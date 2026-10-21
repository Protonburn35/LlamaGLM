#include "stage-expert-bank.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

using namespace glm_stage;

static void require(bool condition, const char * message) { if (!condition) { throw std::runtime_error(message); } }
static void refused(const std::function<void()> & action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid expert bank operation was admitted");
}

constexpr int64_t width = 256, output_width = 16, expert_count = 16, selected_count = 8, tokens = 3;

struct fixture {
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("glm expert bank fixture " + std::to_string(std::random_device{}()));
    std::vector<expert_projection> specs;
    std::array<std::vector<uint8_t>, 3> quant;
    std::shared_ptr<expert_file> file;
    uint64_t selected_bytes = 0;
    fixture(const std::array<ggml_type, 3> & types) {
        require(std::filesystem::create_directory(directory), "cannot create expert bank fixture directory");
        const auto path = directory / "native quantized expert bytes.bin";
        std::ofstream writer(path, std::ios::binary);
        uint64_t offset = 0;
        const std::vector<float> imatrix(width, 1.0f);
        for (size_t p = 0; p < types.size(); ++p) {
            std::vector<float> values(width * output_width * expert_count);
            for (size_t i = 0; i < values.size(); ++i) { values[i] = float(int((i * 13 + p * 17) % 127) - 63) * 0.002f; }
            const uint64_t slab = ggml_row_size(types[p], width) * output_width;
            quant[p].resize(size_t(slab * expert_count));
            if (types[p] == GGML_TYPE_F32) { std::memcpy(quant[p].data(), values.data(), quant[p].size()); }
            else {
                const size_t count = ggml_quantize_chunk(types[p], values.data(), quant[p].data(), 0,
                    output_width * expert_count, width, imatrix.data());
                require(count == quant[p].size(), "native quantizer returned an unexpected size");
            }
            expert_projection spec{types[p], width, output_width, {}};
            for (int64_t e = 0; e < expert_count; ++e) { spec.experts.push_back({0, offset + uint64_t(e) * slab, slab}); }
            specs.push_back(std::move(spec));
            selected_bytes += slab * selected_count;
            writer.write(reinterpret_cast<const char *>(quant[p].data()), quant[p].size());
            offset += quant[p].size();
        }
        writer.close();
        require(bool(writer), "cannot write quantized expert fixture");
        file = std::make_shared<expert_file>(path);
    }
    std::vector<expert_address> catalog() const {
        std::vector<expert_address> result;
        for (const auto & spec : specs) { result.insert(result.end(), spec.experts.begin(), spec.experts.end()); }
        return result;
    }
    ~fixture() { file.reset(); std::error_code error; std::filesystem::remove_all(directory, error); }
};

static void compare_operations(ggml_backend_t backend, const std::array<ggml_tensor *, 3> & reference,
                               const expert_bank::lease & bank, const std::vector<int32_t> & selected) {
    ggml_context_ptr graph(ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true}));
    require(bool(graph), "cannot create expert operation metadata");
    auto * input = ggml_new_tensor_3d(graph.get(), GGML_TYPE_F32, width, 1, tokens);
    auto * original = ggml_new_tensor_2d(graph.get(), GGML_TYPE_I32, selected_count, tokens);
    auto * remapped = ggml_new_tensor_2d(graph.get(), GGML_TYPE_I32, selected_count, tokens);
    auto * scales = ggml_new_tensor_2d(graph.get(), GGML_TYPE_F32, 1, expert_count);
    auto * repeated_scales = ggml_repeat_4d(graph.get(), scales, 1, expert_count, tokens, 1);
    auto * selected_scales = ggml_get_rows(graph.get(), repeated_scales, original);
    std::array<ggml_tensor *, 3> expected, actual;
    auto * computation = ggml_new_graph_custom(graph.get(), 64, false);
    for (size_t p = 0; p < expected.size(); ++p) {
        expected[p] = ggml_mul(graph.get(), ggml_mul_mat_id(graph.get(), reference[p], input, original), selected_scales);
        actual[p] = ggml_mul(graph.get(), ggml_mul_mat_id(graph.get(), bank.projection(p), input, remapped), selected_scales);
        ggml_build_forward_expand(computation, expected[p]);
        ggml_build_forward_expand(computation, actual[p]);
    }
    auto * wrong_scales = ggml_get_rows(graph.get(), repeated_scales, remapped);
    auto * wrong = ggml_mul(graph.get(), ggml_mul_mat_id(graph.get(), bank.projection(0), input, remapped), wrong_scales);
    ggml_build_forward_expand(computation, wrong);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(graph.get(), backend));
    require(bool(buffer), "cannot allocate expert operation inputs/output");
    std::vector<float> inputs(width * tokens);
    for (size_t i = 0; i < inputs.size(); ++i) { inputs[i] = float(int((i * 7) % 31) - 15) * 0.003f; }
    std::array<float, expert_count> scale_values;
    for (size_t i = 0; i < scale_values.size(); ++i) { scale_values[i] = 0.5f + float(i) * 0.03125f; }
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    ggml_backend_tensor_set(original, selected.data(), 0, selected.size() * sizeof(int32_t));
    ggml_backend_tensor_set(remapped, bank.remapped_ids().data(), 0, selected.size() * sizeof(int32_t));
    ggml_backend_tensor_set(scales, scale_values.data(), 0, sizeof(scale_values));
    require(ggml_backend_graph_compute(backend, computation) == GGML_STATUS_SUCCESS, "native expert matrix operations failed");
    std::vector<float> a(output_width * selected_count * tokens), b(a.size());
    for (size_t p = 0; p < expected.size(); ++p) {
        ggml_backend_tensor_get(expected[p], a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(actual[p], b.data(), 0, b.size() * sizeof(float));
        require(std::all_of(a.begin(), a.end(), [](float v) { return std::isfinite(v); }), "full expert reference is not finite");
        require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "materialized quantized expert operation differs from full weights");
    }
    ggml_backend_tensor_get(expected[0], a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(wrong, b.data(), 0, b.size() * sizeof(float));
    require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0, "scale fixture does not distinguish original IDs from cache slots");
}

static void native_quant_case(ggml_backend_t backend, const std::array<ggml_type, 3> & types, uint32_t slots = 8) {
    fixture f(types);
    desktop_experts source(0, {f.file}, f.catalog(), f.selected_bytes, 24);
    uint64_t acquisitions = 0;
    expert_bank bank(backend, slots, f.specs, 2u << 20, [&](auto address) { ++acquisitions; return source.acquire(address); });
    ggml_context_ptr metadata(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
    std::array<ggml_tensor *, 3> reference;
    for (size_t p = 0; p < reference.size(); ++p) { reference[p] = ggml_new_tensor_3d(metadata.get(), types[p], width, output_width, expert_count); }
    ggml_backend_buffer_ptr weights(ggml_backend_alloc_ctx_tensors(metadata.get(), backend));
    require(bool(weights), "cannot allocate independent full expert weights");
    for (size_t p = 0; p < reference.size(); ++p) { ggml_backend_tensor_set(reference[p], f.quant[p].data(), 0, f.quant[p].size()); }
    std::vector<int32_t> selected;
    const std::array<int32_t, 8> odd{15,3,11,7,1,9,5,13};
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t k = 0; k < selected_count; ++k) { selected.push_back(odd[size_t((k + t) % selected_count)]); }
    }
    for (int phase = 0; phase < 6; ++phase) {
        if (phase == 1 || phase == 4) { std::rotate(selected.begin(), selected.begin() + 1, selected.end()); }
        if (phase == 2) { for (auto & id : selected) { --id; } }
        if (phase == 3) { for (auto & id : selected) { if (id == 2) { id = 3; } } }
        if (phase == 5) { for (size_t i = 0; i < selected.size(); ++i) { selected[i] = odd[i % odd.size()]; } }
        const auto before = bank.stats();
        auto lease = bank.materialize(selected);
        refused([&] { bank.materialize(selected); });
        auto competing = std::async(std::launch::async, [&] { refused([&] { bank.materialize(selected); }); });
        competing.get();
        auto moved = std::move(lease);
        refused([&] { lease.projection(0); });
        compare_operations(backend, reference, moved, selected);
        const auto after = bank.stats();
        const uint64_t expected_misses = phase == 0 || phase == 2 ? 8 : phase == 3 ? 1 : phase == 5 ? (slots == 12 ? 4 : 7) : 0;
        require(after.materializations == uint64_t(phase + 1) && after.failures == 0 &&
                after.slot_misses - before.slot_misses == expected_misses && after.slot_hits - before.slot_hits == 8 - expected_misses &&
                after.copied_slabs - before.copied_slabs == 3 * expected_misses && after.copied_slabs == acquisitions,
                "backend expert reuse/LRU copied hot experts, lost selected slots or reported incorrect counters");
        if (expected_misses == 0) { require(after.copied_bytes == before.copied_bytes, "hot bank selection copied backend bytes"); }
        if (phase == 0) { require(f.file->stats().requests == 24, "first bank fill did not read all eight experts of all three projections"); }
        if (phase == 1) { require(f.file->stats().requests == 24, "hot bank fill reread source files"); }
        if (phase == 2) { require(f.file->stats().requests == 48, "cold replacement did not fetch the complete new expert set"); }
        if (phase == 3 || phase == 4) { require(f.file->stats().requests == 51, "partial/hot replacement fetched unchanged backend experts"); }
        if (phase == 5) { require(f.file->stats().requests == (slots == 12 ? 63 : 72), "returning experts did not follow retained backend slots"); }
    }
    std::printf("GLM expert bank: %s/%s/%s, %u slots, native CPU outputs byte-exact across cold/hot/partial/LRU phases; hot copies zero\n",
                ggml_type_name(types[0]), ggml_type_name(types[1]), ggml_type_name(types[2]), slots);
}

static void failure_cases(ggml_backend_t backend) {
    fixture f({GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32});
    memory_experts memory(f.file->size(), 48);
    for (size_t p = 0; p < f.specs.size(); ++p) {
        for (size_t e = 0; e < f.specs[p].experts.size(); ++e) {
            const auto address = f.specs[p].experts[e];
            memory.preload(address, f.quant[p].data() + e * size_t(address.bytes), size_t(address.bytes));
        }
    }
    memory.seal(f.catalog());
    size_t calls = 0;
    const auto source = [&](expert_address a) { ++calls; return memory.acquire(a); };
    refused([&] { expert_bank invalid(backend, 8, f.specs, 1, source); });
    require(calls == 0, "invalid bank allocation reached the source provider");
    for (int mode = 0; mode < 3; ++mode) {
        expert_bank bank(backend, 8, f.specs, 1u << 20, source);
        std::vector<int32_t> ids(8);
        std::iota(ids.begin(), ids.end(), 0);
        if (mode == 0) { ids[0] = -1; }
        if (mode == 1) { ids[0] = 16; }
        if (mode == 2) { ids.push_back(8); }
        refused([&] { bank.materialize(ids); });
        refused([&] { bank.materialize({0,1,2,3,4,5,6,7}); });
        require(calls == 0, "invalid selected expert IDs reached materialization");
    }
    expert_bank broken(backend, 8, f.specs, 1u << 20, [&](expert_address address) -> expert_lease {
        ++calls;
        if (calls < 3) { return memory.acquire(address); }
        return std::make_shared<const std::vector<uint8_t>>(1);
    });
    refused([&] { broken.materialize({0,1,2,3,4,5,6,7}); });
    refused([&] { broken.materialize({0,1,2,3,4,5,6,7}); });
    require(calls == 3, "partly failed bank was reused");
    size_t replacement_calls = 0;
    bool fail_replacement = false;
    expert_bank replacing(backend, 8, f.specs, 1u << 20, [&](expert_address address) -> expert_lease {
        ++replacement_calls;
        if (fail_replacement && replacement_calls == 26) { return std::make_shared<const std::vector<uint8_t>>(1); }
        return memory.acquire(address);
    });
    { auto first = replacing.materialize({0,1,2,3,4,5,6,7}); }
    fail_replacement = true;
    refused([&] { replacing.materialize({1,2,3,4,5,6,7,8}); });
    refused([&] { replacing.materialize({0,1,2,3,4,5,6,7}); });
    const auto stats = replacing.stats();
    require(replacement_calls == 26 && stats.materializations == 1 && stats.copied_slabs == 25 && stats.failures == 2,
            "partly failed hot-bank replacement exposed/reused old slot mappings or lost completed-copy accounting");
}

int main() {
    try {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        require(bool(backend), "CPU backend is unavailable");
        ggml_backend_cpu_set_n_threads(backend.get(), 2);
        native_quant_case(backend.get(), {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32});
        native_quant_case(backend.get(), {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_S});
        native_quant_case(backend.get(), {GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_Q3_K});
        native_quant_case(backend.get(), {GGML_TYPE_Q2_K, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S});
        native_quant_case(backend.get(), {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32}, 12);
        failure_cases(backend.get());
        std::puts("GLM expert bank: six suites passed; all eight publisher types, eight selected experts, backend reuse/LRU, original scale IDs and failure refusal");
        ggml_quantize_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM expert bank test failed: %s\n", error.what());
        return 1;
    }
}
