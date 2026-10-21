#include "stage-host-buffer.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>

namespace glm_stage {

struct host_buffer_pool::impl : std::enable_shared_from_this<host_buffer_pool::impl> {
    struct data_buffer {
        std::shared_ptr<impl> owner;
        std::unique_ptr<locked_pages> pages;
        ggml_backend_buffer_ptr cpu;
        size_t requested = 0;
        uint64_t rounded = 0;
        bool registered = false, charged = false;
        ~data_buffer() {
            if (registered) { std::lock_guard<std::mutex> lock(owner->mutex); owner->active.erase(this); }
            cpu.reset();
            pages.reset();
            if (charged) { owner->release(requested, rounded); }
        }
    };
    struct reservation {
        std::shared_ptr<impl> owner;
        size_t requested;
        uint64_t rounded;
        bool pending = true;
        ~reservation() { if (pending) { owner->release(requested, rounded); } }
    };
    uint64_t budget;
    size_t buffer_budget;
    ggml_backend_buffer_type type{};
    mutable std::mutex mutex;
    std::set<data_buffer *> active;
    host_buffer_stats counters{};
    std::atomic<uint64_t> allocations{0}, failures{0};
    std::atomic<int> last_native_error{0};

    static data_buffer & data(ggml_backend_buffer_t buffer) { return *static_cast<data_buffer *>(buffer->context); }
    static void free_buffer(ggml_backend_buffer_t buffer) { delete &data(buffer); }
    static void * base(ggml_backend_buffer_t buffer) { return ggml_backend_buffer_get_base(data(buffer).cpu.get()); }
    static ggml_status init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
        auto * cpu = data(buffer).cpu.get();
        return cpu->iface.init_tensor ? cpu->iface.init_tensor(cpu, tensor) : GGML_STATUS_SUCCESS;
    }
    static void memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t bytes) {
        if (bytes == 0) { return; }
        auto * cpu = data(buffer).cpu.get(); cpu->iface.memset_tensor(cpu, tensor, value, offset, bytes);
    }
    static void set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * source, size_t offset, size_t bytes) {
        if (bytes == 0) { return; }
        auto * cpu = data(buffer).cpu.get(); cpu->iface.set_tensor(cpu, tensor, source, offset, bytes);
    }
    static void get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * output, size_t offset, size_t bytes) {
        if (bytes == 0) { return; }
        auto * cpu = data(buffer).cpu.get(); cpu->iface.get_tensor(cpu, tensor, output, offset, bytes);
    }
    static bool copy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * output) {
        if (ggml_nbytes(output) == 0) { return true; }
        auto * cpu = data(buffer).cpu.get();
        return cpu->iface.cpy_tensor && cpu->iface.cpy_tensor(cpu, source, output);
    }
    static void clear(ggml_backend_buffer_t buffer, uint8_t value) {
        auto * cpu = data(buffer).cpu.get(); cpu->iface.clear(cpu, value);
    }
    static const char * name(ggml_backend_buffer_type_t) { return "CPU_Locked"; }
    static bool is_host(ggml_backend_buffer_type_t) { return true; }
    static size_t alignment(ggml_backend_buffer_type_t) { return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type()); }
    static size_t allocation_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
        return ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), tensor);
    }
    static size_t max_size(ggml_backend_buffer_type_t type) {
        return size_t(std::min<uint64_t>(static_cast<impl *>(type->context)->budget, std::numeric_limits<size_t>::max()));
    }
    static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t bytes) noexcept {
        auto & state = *static_cast<impl *>(type->context);
        try {
            const auto owner = state.shared_from_this();
            const uint64_t rounded = locked_pages::required_bytes(bytes);
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                if (rounded > state.budget || state.counters.reserved_page_bytes > state.budget - rounded ||
                        state.counters.reserved_buffers >= state.buffer_budget) {
                    ++state.counters.refusals;
                    return nullptr;
                }
                state.counters.reserved_page_bytes += rounded;
                state.counters.requested_bytes += bytes;
                ++state.counters.reserved_buffers;
                state.counters.peak_reserved_page_bytes = std::max(state.counters.peak_reserved_page_bytes, state.counters.reserved_page_bytes);
            }
            reservation reserved{owner, bytes, rounded};
            auto value = std::make_unique<data_buffer>();
            value->owner = owner; value->requested = bytes; value->rounded = rounded;
            value->pages = std::make_unique<locked_pages>(bytes, rounded);
            if (uintptr_t(value->pages->data()) % alignment(type)) { throw std::runtime_error("native host pages do not meet CPU alignment"); }
            value->cpu.reset(ggml_backend_cpu_buffer_from_ptr(value->pages->data(), bytes));
            if (!value->cpu) { throw std::bad_alloc(); }
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.active.insert(value.get()); value->registered = true;
            }
            const ggml_backend_buffer_i callbacks{free_buffer, base, init_tensor, memset_tensor, set_tensor, get_tensor,
                nullptr, nullptr, copy_tensor, clear, nullptr};
            auto * result = ggml_backend_buffer_init(type, callbacks, value.get(), bytes);
            if (!result) { throw std::bad_alloc(); }
            value->charged = true; reserved.pending = false;
            value.release();
            ++state.allocations;
            return result;
        } catch (const std::system_error & error) {
            state.last_native_error.store(error.code().value());
            ++state.failures;
            return nullptr;
        } catch (...) {
            ++state.failures;
            return nullptr;
        }
    }
    void release(size_t requested, uint64_t rounded) {
        std::lock_guard<std::mutex> lock(mutex);
        counters.reserved_page_bytes -= rounded; counters.requested_bytes -= requested; --counters.reserved_buffers;
    }
    impl(ggml_backend_dev_t cpu, uint64_t page_budget, size_t count) : budget(page_budget), buffer_budget(count) {
        if (!cpu || ggml_backend_dev_type(cpu) != GGML_BACKEND_DEVICE_TYPE_CPU ||
                budget == 0 || budget > max_locked_buffer_bytes || count == 0 || count > 4096) {
            throw std::invalid_argument("invalid locked host buffer device/page/entry budget");
        }
        type = {{name, allocate, nullptr, alignment, max_size, allocation_size, nullptr, is_host}, cpu, this};
    }
};

host_buffer_pool::host_buffer_pool(ggml_backend_dev_t cpu, uint64_t page_budget, size_t buffer_budget) :
    state(std::make_shared<impl>(cpu, page_budget, buffer_budget)) {}
ggml_backend_buffer_type_t host_buffer_pool::buffer_type() const { return &state->type; }
host_buffer_stats host_buffer_pool::stats() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto result = state->counters;
    result.pending_buffers = result.reserved_buffers - state->active.size();
    result.allocations = state->allocations.load(); result.failures = state->failures.load();
    result.last_native_error = state->last_native_error.load();
    return result;
}
void host_buffer_pool::verify_all() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->counters.reserved_buffers != state->active.size()) { throw std::runtime_error("host buffers have unfinished allocation/free work"); }
    for (const auto * buffer : state->active) { buffer->pages->verify_locked(); }
}

} // namespace glm_stage
