#include "llama-stage-provider.h"
#include "llama-model.h"

#include <cstring>
#include <stdexcept>

int llama_stage_expert_projection(const LLM_TN_IMPL & name) {
    if (name.bid < 0 || !name.suffix || std::strcmp(name.suffix, "weight")) { return -1; }
    switch (name.tensor) {
        case LLM_TENSOR_FFN_GATE_EXPS: return 0;
        case LLM_TENSOR_FFN_UP_EXPS:   return 1;
        case LLM_TENSOR_FFN_DOWN_EXPS: return 2;
        default: return -1;
    }
}

#ifdef LLAMA_GLM_STAGE_STORAGE
#include "glm-stage/stage-expert-bank.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>

namespace {

struct stage_layer_source {
    std::array<ggml_tensor *, 3> logical{};
    std::array<glm_stage::expert_projection, 3> specs{};
};

struct stage_weight_source;

struct stage_context_experts final : llama_stage_context_experts {
    struct layer_context {
        stage_context_experts * owner;
        std::unique_ptr<glm_stage::expert_bank> bank;
        std::optional<glm_stage::expert_bank::lease> active;
    };
    std::shared_ptr<stage_weight_source> source;
    std::map<int, std::unique_ptr<layer_context>> layers;
    std::map<const ggml_tensor *, std::pair<layer_context *, size_t>> weights;
    std::map<ggml_backend_buffer_type_t, size_t> buffers;
    std::atomic<bool> invalid{false};
    mutable std::mutex error_mutex;
    std::exception_ptr failure;

    stage_context_experts(std::shared_ptr<stage_weight_source> source, const std::vector<ggml_backend_t> & backends);
    bool owns(const ggml_tensor * tensor) const override { return weights.count(tensor) != 0; }
    ggml_tensor * projection(const ggml_tensor * tensor) const override {
        const auto & weight = weights.at(tensor);
        return weight.first->bank->projection_for_graph(weight.second);
    }
    bool failed() const override { return invalid.load(); }
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return buffers; }
    std::string error() const override {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!failure) { return "unknown expert provider failure"; }
        try { std::rethrow_exception(failure); }
        catch (const std::exception & e) { return e.what(); }
        catch (...) { return "non-standard expert provider failure"; }
    }
    void record_failure() noexcept {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!invalid.load()) { failure = std::current_exception(); invalid.store(true); }
    }
    static void materialize(ggml_tensor * dst, const ggml_tensor * src, int ith, int, void * userdata) noexcept {
        if (ith != 0) { return; }
        auto & layer = *static_cast<layer_context *>(userdata);
        std::memset(dst->data, 0, ggml_nbytes(dst));
        try {
            if (layer.owner->failed()) { return; }
            if (src->type != GGML_TYPE_I32 || !ggml_is_contiguous(src) || ggml_nelements(src) <= 0 || ggml_nelements(src) > 16384) {
                throw std::runtime_error("invalid routed expert IDs at the stage boundary");
            }
            const auto * ids = static_cast<const int32_t *>(src->data);
            std::vector<int32_t> selected(ids, ids + ggml_nelements(src));
            layer.active.reset();
            layer.active.emplace(layer.bank->materialize(selected));
            const auto & remapped = layer.active->remapped_ids();
            std::memcpy(dst->data, remapped.data(), remapped.size() * sizeof(int32_t));
        } catch (...) { layer.owner->record_failure(); }
    }
    ggml_tensor * build_slots(ggml_context * ctx, ggml_tensor * selected, int layer) override {
        const auto found = layers.find(layer);
        if (found == layers.end()) { throw std::runtime_error("streamed MoE layer has no owned expert bank"); }
        auto * contiguous = ggml_is_contiguous(selected) ? selected : ggml_cont(ctx, selected);
        return ggml_map_custom1(ctx, contiguous, materialize, 1, found->second.get());
    }
};

struct stage_weight_source final : llama_stage_weight_source, std::enable_shared_from_this<stage_weight_source> {
    const llama_model & model;
    llama_stage_storage_params params;
    ggml_context_ptr metadata;
    std::map<int, stage_layer_source> layers;
    std::vector<std::shared_ptr<glm_stage::expert_file>> files;
    std::shared_ptr<glm_stage::desktop_experts> cache;
    bool finalized = false;

    stage_weight_source(const llama_model & model, const llama_stage_storage_params & params) : model(model), params(params) {
        if (!model.stage_enabled || model.stage.rank == 1 || model.arch != LLM_ARCH_GLM5_NEXT ||
                model.n_devices() > 1 || model.split_mode() == LLAMA_SPLIT_MODE_TENSOR ||
                !model.hdmd.weight_blocks.empty() || !model.hdmd.inverse_blocks.empty() ||
                model.hdmd.gdn_v_grouped || model.hdmd.tied_output ||
                params.bank_slots < model.hparams.n_expert_used_max() || params.bank_slots > model.hparams.n_expert) {
            throw std::invalid_argument("unsupported desktop stage storage geometry/device/transform policy");
        }
        const uint32_t begin = model.stage.rank == 0 ? 0 : model.stage.layer_ends[model.stage.rank - 1];
        const uint32_t end = model.stage.layer_ends[model.stage.rank];
        const uint32_t count = 3 * (end - begin);
        metadata.reset(ggml_init({size_t(count) * ggml_tensor_overhead(), nullptr, true}));
        if (!metadata) { throw std::runtime_error("cannot allocate streamed tensor metadata"); }
    }
    ggml_tensor * register_tensor(const LLM_TN_IMPL & name, const ggml_tensor * descriptor, uint16_t file, uint64_t offset) override {
        const int projection = llama_stage_expert_projection(name);
        if (finalized || projection < 0 || !descriptor || !model.stage_owns_layer(name.bid) ||
                !glm_stage::expert_bank::supports_type(descriptor->type) || descriptor->ne[2] != model.hparams.n_expert ||
                descriptor->ne[3] != 1 || !ggml_is_contiguous(descriptor)) {
            throw std::invalid_argument("invalid owned streamed expert tensor");
        }
        auto & layer = layers[name.bid];
        if (layer.logical[projection]) { throw std::invalid_argument("duplicate streamed expert projection"); }
        auto * tensor = ggml_dup_tensor(metadata.get(), descriptor);
        ggml_set_name(tensor, name.str().c_str());
        layer.logical[projection] = tensor;
        auto & spec = layer.specs[projection];
        spec.type = tensor->type; spec.input = tensor->ne[0]; spec.output = tensor->ne[1];
        const uint64_t bytes = tensor->nb[2];
        if (bytes == 0 || bytes > glm_stage::max_expert_read_bytes || bytes > params.host_cache_bytes ||
                offset > std::numeric_limits<uint64_t>::max() - ggml_nbytes(tensor)) {
            throw std::invalid_argument("streamed expert slab exceeds bounded source/cache sizes");
        }
        for (int64_t expert = 0; expert < tensor->ne[2]; ++expert) { spec.experts.push_back({file, offset + uint64_t(expert) * bytes, bytes}); }
        return tensor;
    }
    void finalize(const std::vector<std::string> & paths) override {
        if (finalized) { throw std::runtime_error("streamed model source was finalized twice"); }
        const uint32_t begin = model.stage.rank == 0 ? 0 : model.stage.layer_ends[model.stage.rank - 1];
        const uint32_t end = model.stage.layer_ends[model.stage.rank];
        std::vector<glm_stage::expert_address> catalog;
        for (uint32_t layer = begin; layer < end; ++layer) {
            if (layer < model.hparams.n_layer_dense_lead) { continue; }
            const auto found = layers.find(int(layer));
            if (found == layers.end()) { throw std::runtime_error("owned MoE layer lacks streamed descriptors"); }
            for (size_t p = 0; p < 3; ++p) {
                if (!found->second.logical[p]) { throw std::runtime_error("owned MoE layer lacks a routed projection"); }
                const auto & experts = found->second.specs[p].experts;
                catalog.insert(catalog.end(), experts.begin(), experts.end());
            }
        }
        if (!catalog.empty()) {
            if (paths.empty() || paths.size() > 100) { throw std::invalid_argument("streamed stages require local named GGUF shards"); }
            for (const auto & path : paths) {
                if (path.empty()) { throw std::invalid_argument("streamed GGUF shard path is absent"); }
                files.push_back(std::make_shared<glm_stage::expert_file>(std::filesystem::u8path(path)));
            }
            cache = std::make_shared<glm_stage::desktop_experts>(model.stage.rank, files, catalog,
                params.host_cache_bytes, params.host_cache_entries, params.max_active_reads);
        }
        finalized = true;
    }
    void append_tensors(std::vector<std::pair<std::string, ggml_tensor *>> & output) const override {
        for (const auto & layer : layers) {
            for (auto * tensor : layer.second.logical) { output.emplace_back(ggml_get_name(tensor), tensor); }
        }
    }
    bool empty() const override { return layers.empty(); }
    uint32_t slots() const override { return params.bank_slots; }
    void report(llama_stage_storage_report & output) const override {
        output.streamed_tensors = uint32_t(layers.size() * 3);
        for (const auto & layer : layers) { for (const auto tensor : layer.second.logical) { output.routed_logical_bytes += ggml_nbytes(tensor); } }
        if (cache) { output.host_reserved_bytes = cache->stats().reserved_payload_bytes; }
        for (const auto & file : files) {
            const auto stats = file->stats();
            output.native_read_requests += stats.requests;
            output.native_system_reads += stats.system_reads;
            output.native_read_bytes += stats.bytes;
        }
    }
    std::unique_ptr<llama_stage_context_experts> create_context(const std::vector<ggml_backend_t> & backends) override {
        if (!finalized) { throw std::runtime_error("streamed source is not finalized"); }
        if (empty()) { return nullptr; }
        return std::make_unique<stage_context_experts>(shared_from_this(), backends);
    }
};

stage_context_experts::stage_context_experts(std::shared_ptr<stage_weight_source> source, const std::vector<ggml_backend_t> & backends) :
    source(std::move(source)) {
    uint64_t remaining = this->source->params.context_bank_bytes;
    for (const auto & registered : this->source->layers) {
        ggml_backend_t backend = nullptr;
        for (const auto candidate : backends) {
            if (ggml_backend_get_device(candidate) == this->source->model.dev_layer(registered.first)) { backend = candidate; break; }
        }
        if (!backend) { throw std::runtime_error("owned streamed layer has no matching execution backend"); }
        auto layer = std::make_unique<layer_context>();
        layer->owner = this;
        const auto cache = this->source->cache;
        const auto & specs = registered.second.specs;
        layer->bank = std::make_unique<glm_stage::expert_bank>(backend, this->source->params.bank_slots,
            std::vector<glm_stage::expert_projection>(specs.begin(), specs.end()), std::min<uint64_t>(remaining, 1ull << 30),
            [cache](glm_stage::expert_address address) { return cache->acquire(address); });
        const uint64_t bytes = layer->bank->buffer_bytes();
        remaining -= bytes;
        buffers[ggml_backend_get_default_buffer_type(backend)] += bytes;
        for (size_t projection = 0; projection < 3; ++projection) {
            ggml_set_name(layer->bank->projection_for_graph(projection), ggml_get_name(registered.second.logical[projection]));
            weights.emplace(registered.second.logical[projection], std::make_pair(layer.get(), projection));
        }
        layers.emplace(registered.first, std::move(layer));
    }
}

} // namespace

std::shared_ptr<llama_stage_weight_source> llama_stage_make_weight_source(
        const llama_model & model, const llama_stage_storage_params & params) {
    return std::make_shared<stage_weight_source>(model, params);
}
#else
std::shared_ptr<llama_stage_weight_source> llama_stage_make_weight_source(
        const llama_model &, const llama_stage_storage_params &) {
    throw std::runtime_error("streamed stages require a build with LLAMA_GLM_RPC enabled");
}
#endif
