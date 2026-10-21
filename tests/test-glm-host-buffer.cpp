#include "stage-host-buffer.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <vector>

using namespace glm_stage;

static void require(bool condition, const char * message) { if (!condition) { throw std::runtime_error(message); } }
template<class F> static void refused(F function) {
    try { function(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid host buffer policy was accepted");
}

struct matrix {
    ggml_type type;
    std::vector<uint8_t> weights;
    std::vector<float> input;
    explicit matrix(ggml_type type) : type(type), weights(ggml_row_size(type, 256) * 4), input(256 * 2) {
        std::vector<float> values(256 * 4), importance(256, 1.0f);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = float(int((i * 13) % 61) - 30) * 0.01f; }
        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int((i * 7) % 29) - 14) * 0.02f; }
        if (type == GGML_TYPE_F32) { std::memcpy(weights.data(), values.data(), weights.size()); }
        else {
            require(ggml_quantize_chunk(type, values.data(), weights.data(), 0, 4, 256, importance.data()) == weights.size(),
                    "host matrix quantizer returned an unexpected size");
        }
    }
};

struct graph {
    ggml_context_ptr metadata{ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true})};
    ggml_tensor * weights;
    ggml_tensor * input;
    ggml_tensor * output;
    ggml_cgraph * computation;
    explicit graph(ggml_type type) {
        require(bool(metadata), "cannot create host operation metadata");
        weights = ggml_new_tensor_2d(metadata.get(), type, 256, 4);
        input = ggml_new_tensor_2d(metadata.get(), GGML_TYPE_F32, 256, 2);
        output = ggml_mul_mat(metadata.get(), weights, input);
        ggml_set_input(weights); ggml_set_input(input); ggml_set_output(output);
        computation = ggml_new_graph_custom(metadata.get(), 32, false);
        ggml_build_forward_expand(computation, output);
    }
    void set(const matrix & values) {
        ggml_backend_tensor_set(weights, values.weights.data(), 0, values.weights.size());
        ggml_backend_tensor_set(input, values.input.data(), 0, values.input.size() * sizeof(float));
    }
    std::vector<float> get() {
        std::vector<float> values(8);
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        return values;
    }
};

static std::vector<float> direct(ggml_backend_t backend, const matrix & values, host_buffer_pool * pool = nullptr) {
    graph operation(values.type);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(operation.metadata.get(),
        pool ? pool->buffer_type() : ggml_backend_get_default_buffer_type(backend)));
    require(bool(buffer), "cannot allocate CPU matrix buffers");
    ggml_backend_buffer_clear(buffer.get(), 0);
    operation.set(values);
    if (pool) { pool->verify_all(); }
    require(ggml_backend_graph_compute(backend, operation.computation) == GGML_STATUS_SUCCESS, "host matrix computation failed");
    if (pool) { pool->verify_all(); }
    return operation.get();
}

static void quantized_operations(ggml_backend_t backend) {
    const std::array<ggml_type, 8> types{GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K};
    for (const auto type : types) {
        host_buffer_pool pool(ggml_backend_get_device(backend), 16u << 10);
        matrix values(type);
        const auto expected = direct(backend, values);
        const auto actual = direct(backend, values, &pool);
        require(std::all_of(expected.begin(), expected.end(), [](float value) { return std::isfinite(value); }) &&
                std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0,
                "locked host matrix differs from ordinary CPU buffers");
        const auto stats = pool.stats();
        require(stats.allocations == 1 && stats.reserved_page_bytes == 0 && stats.reserved_buffers == 0 &&
                stats.pending_buffers == 0 && stats.peak_reserved_page_bytes > 0 && stats.peak_reserved_page_bytes <= (16u << 10) &&
                stats.failures == 0 && stats.refusals == 0, "host matrix allocation accounting differs");
    }
    std::puts("GLM host buffers: all eight publisher types have byte-exact native CPU matrix outputs");
}

static void scheduler(ggml_backend_t backend) {
    host_buffer_pool pool(ggml_backend_get_device(backend), 32u << 10);
    matrix values(GGML_TYPE_F32);
    const auto expected = direct(backend, values);
    {
        graph operation(values.type);
        auto * buft = pool.buffer_type();
        ggml_backend_sched_ptr schedule(ggml_backend_sched_new(&backend, &buft, 1, 32, false, false));
        require(bool(schedule) && ggml_backend_sched_reserve(schedule.get(), operation.computation) &&
                ggml_backend_sched_alloc_graph(schedule.get(), operation.computation), "locked CPU scheduler reserve failed");
        for (const auto * tensor : {operation.weights, operation.input, operation.output}) {
            require(tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer) == buft,
                    "CPU scheduler silently allocated a pageable tensor buffer");
        }
        operation.set(values);
        pool.verify_all();
        require(ggml_backend_sched_graph_compute(schedule.get(), operation.computation) == GGML_STATUS_SUCCESS,
                "locked scheduler computation failed");
        const auto actual = operation.get();
        require(std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0,
                "locked scheduler changed matrix results");
        pool.verify_all();
    }
    require(pool.stats().reserved_page_bytes == 0 && pool.stats().reserved_buffers == 0, "scheduler destruction retained native pages");
}

static void capacity(ggml_backend_dev_t cpu) {
    const auto page = locked_pages::required_bytes(1);
    const auto baseline = locked_pages::locked_stats();
    refused([&] { host_buffer_pool invalid(nullptr, page); });
    refused([&] { host_buffer_pool invalid(cpu, 0); });
    refused([&] { host_buffer_pool invalid(cpu, page, 0); });
    host_buffer_pool pool(cpu, 2 * page, 2);
    auto * buft = pool.buffer_type();
    require(ggml_backend_buft_is_host(buft) && ggml_backend_dev_supports_buft(cpu, buft), "CPU rejects the locked standard buffer layout");
    {
        ggml_backend_buffer_ptr dummy(ggml_backend_buft_alloc_buffer(buft, 0));
        require(dummy && ggml_backend_buffer_get_size(dummy.get()) == 0 && pool.stats().allocations == 0,
                "GGML dummy buffer unexpectedly allocated payload pages");
    }
    ggml_backend_buffer_ptr first(ggml_backend_buft_alloc_buffer(buft, 1)), second(ggml_backend_buft_alloc_buffer(buft, 1));
    require(first && second, "cannot allocate bounded host buffers");
    ggml_backend_buffer_ptr full(ggml_backend_buft_alloc_buffer(buft, 1));
    require(!full && pool.stats().reserved_page_bytes == 2 * page && pool.stats().reserved_buffers == 2,
            "host allocation exceeded its shared page/entry capacity");
    pool.verify_all();
    first.reset(); second.reset();
    ggml_backend_buffer_ptr rounded(ggml_backend_buft_alloc_buffer(buft, size_t(page + 1)));
    require(rounded && pool.stats().reserved_page_bytes == 2 * page && pool.stats().requested_bytes == page + 1,
            "host allocator did not charge page rounding");
    ggml_backend_buffer_clear(rounded.get(), 0x5a);
    require(static_cast<const uint8_t *>(ggml_backend_buffer_get_base(rounded.get()))[page] == 0x5a, "host clear/base delegation differs");
    rounded.reset();
    {
        host_buffer_pool limited(cpu, page, 2);
        ggml_context_ptr metadata(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
        ggml_new_tensor_1d(metadata.get(), GGML_TYPE_F32, page / sizeof(float));
        ggml_new_tensor_1d(metadata.get(), GGML_TYPE_F32, page / sizeof(float));
        ggml_backend_buffer_ptr rejected(ggml_backend_alloc_ctx_tensors_from_buft(metadata.get(), limited.buffer_type()));
        require(!rejected && limited.stats().allocations == 1 && limited.stats().refusals == 1 &&
                limited.stats().reserved_page_bytes == 0 && limited.stats().reserved_buffers == 0,
                "partly allocated GGML context leaked pages or fell back outside the budget");
    }
    const auto after = locked_pages::locked_stats();
    require(after.bytes == baseline.bytes && after.buffers == baseline.buffers, "host capacity tests leaked native locks");
}

static void concurrent_lifetimes(ggml_backend_dev_t cpu) {
    const auto page = locked_pages::required_bytes(1);
    const auto baseline = locked_pages::locked_stats();
    {
        host_buffer_pool pool(cpu, 4 * page, 4);
        std::promise<void> start;
        const auto ready = start.get_future().share();
        std::vector<std::future<ggml_backend_buffer_ptr>> workers;
        for (int i = 0; i < 8; ++i) {
            workers.push_back(std::async(std::launch::async, [&] {
                ready.wait();
                return ggml_backend_buffer_ptr(ggml_backend_buft_alloc_buffer(pool.buffer_type(), 1));
            }));
        }
        start.set_value();
        std::vector<ggml_backend_buffer_ptr> retained;
        for (auto & worker : workers) { auto buffer = worker.get(); if (buffer) { retained.push_back(std::move(buffer)); } }
        const auto stats = pool.stats();
        require(retained.size() == 4 && stats.reserved_page_bytes == 4 * page && stats.reserved_buffers == 4 &&
                stats.pending_buffers == 0 && stats.refusals == 4 && stats.failures == 0,
                "concurrent host allocations oversubscribed shared capacity");
        pool.verify_all();
    }
    ggml_backend_buffer_ptr retained;
    {
        host_buffer_pool pool(cpu, page, 1);
        retained.reset(ggml_backend_buft_alloc_buffer(pool.buffer_type(), 1));
        require(bool(retained), "cannot allocate retained host buffer");
    }
    require(locked_pages::locked_stats().bytes == baseline.bytes + page && std::strcmp(ggml_backend_buffer_name(retained.get()), "CPU_Locked") == 0,
            "nonempty buffer lost its allocator/lock after wrapper destruction");
    ggml_backend_buffer_clear(retained.get(), 0x37);
    require(*static_cast<const uint8_t *>(ggml_backend_buffer_get_base(retained.get())) == 0x37, "retained native buffer changed bytes");
    retained.reset();
    require(locked_pages::locked_stats().bytes == baseline.bytes && locked_pages::locked_stats().buffers == baseline.buffers,
            "final host buffer release leaked pages");
}

static void external_pages(ggml_backend_dev_t cpu) {
    const auto page = locked_pages::required_bytes(1);
    const auto baseline = locked_pages::locked_stats();
    const auto verify = +[](const void * pointer) -> uint64_t {
        const auto & pages = *static_cast<const locked_pages *>(pointer);
        pages.verify_locked(); return pages.locked_bytes();
    };
    std::weak_ptr<void> weak;
    ggml_backend_buffer_ptr retained;
    {
        host_buffer_pool pool(cpu, 2 * page);
        refused([&] { pool.preload_external_pages(page + 1, [] { return std::shared_ptr<void>{}; }, verify); });
        refused([&] { pool.preload_external_pages(page, [] { return std::shared_ptr<void>{}; }, verify); });
        refused([&] { pool.preload_external_pages(page, [&]() -> std::shared_ptr<void> {
            auto value = std::make_shared<locked_pages>(size_t(page), page);
            throw std::runtime_error("injected startup load failure");
        }, verify); });
        refused([&] { pool.preload_external_pages(page, [&] {
            return std::make_shared<locked_pages>(size_t(2 * page), 2 * page);
        }, verify); });
        require(pool.stats().reserved_page_bytes == 0 && !pool.stats().external_pending &&
                locked_pages::locked_stats().bytes == baseline.bytes, "failed external preload leaked pages or reservations");
        auto owner = pool.preload_external_pages(page, [&] {
            const auto stats = pool.stats();
            require(stats.reserved_page_bytes == page && stats.external_page_bytes == page && stats.external_pending,
                    "external loader ran before its complete page reservation");
            refused([&] { pool.verify_all(); });
            refused([&] { pool.preload_external_pages(page, [] { return std::shared_ptr<void>{}; }, verify); });
            retained = std::async(std::launch::async, [&] {
                return ggml_backend_buffer_ptr(ggml_backend_buft_alloc_buffer(pool.buffer_type(), 1));
            }).get();
            require(bool(retained), "pending external reservation prevented an in-budget concurrent tensor allocation");
            return std::make_shared<locked_pages>(size_t(page), page);
        }, verify);
        weak = owner; owner.reset();
        ggml_backend_buffer_ptr rejected(ggml_backend_buft_alloc_buffer(pool.buffer_type(), 1));
        require(retained && !rejected && pool.stats().reserved_page_bytes == 2 * page &&
                pool.stats().external_page_bytes == page && pool.stats().reserved_buffers == 1,
                "GGML and external pages exceeded their shared budget");
        pool.verify_all();
    }
    require(!weak.expired() && locked_pages::locked_stats().bytes == baseline.bytes + 2 * page,
            "retained tensor buffer lost its external page owner");
    retained.reset();
    require(weak.expired() && locked_pages::locked_stats().bytes == baseline.bytes &&
            locked_pages::locked_stats().buffers == baseline.buffers, "external owner outlived final buffer deletion");
}

int main() {
    try {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        require(bool(backend), "CPU backend is unavailable");
        ggml_backend_cpu_set_n_threads(backend.get(), 2);
        quantized_operations(backend.get()); scheduler(backend.get());
        capacity(ggml_backend_get_device(backend.get())); concurrent_lifetimes(ggml_backend_get_device(backend.get()));
        external_pages(ggml_backend_get_device(backend.get()));
        ggml_quantize_free();
        std::puts("GLM host buffers: five suites passed (native quant ops, scheduler, shared capacity/cleanup, concurrent/retained lifetime, external preload); full admission NOT_VALIDATED");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM host buffer test failed: %s\n", error.what());
        return 1;
    }
}
