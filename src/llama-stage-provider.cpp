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

ggml_backend_buffer_type_t llama_model::stage_buffer_type(ggml_backend_buffer_type_t original) const {
    return stage_source ? stage_source->host_buffer_type(original) : original;
}

#ifdef LLAMA_GLM_STAGE_STORAGE
#include "glm-stage/stage-expert-bank.h"
#include "glm-stage/stage-host-buffer.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <set>

namespace {

struct stage_layer_source {
    std::array<ggml_tensor *, 3> logical{};
    std::array<glm_stage::expert_projection, 3> specs{};
};

struct stage_expert_reader {
    virtual ~stage_expert_reader() = default;
    virtual glm_stage::expert_lease acquire(glm_stage::expert_address address) = 0;
    virtual void report(llama_stage_storage_report & output) const = 0;
    virtual bool report_preload(llama_stage_preload_report &) const { return false; }
};

struct desktop_reader final : stage_expert_reader {
    std::vector<std::shared_ptr<glm_stage::expert_file>> files;
    std::shared_ptr<glm_stage::desktop_experts> cache;
    desktop_reader(uint32_t rank, std::vector<std::shared_ptr<glm_stage::expert_file>> sources,
            const std::vector<glm_stage::expert_address> & catalog, const llama_stage_storage_params & params) : files(std::move(sources)) {
        cache = std::make_shared<glm_stage::desktop_experts>(rank, files, catalog,
            params.host_cache_bytes, params.host_cache_entries, params.max_active_reads);
    }
    glm_stage::expert_lease acquire(glm_stage::expert_address address) override { return cache->acquire(address); }
    void report(llama_stage_storage_report & output) const override {
        output.host_reserved_bytes = cache->stats().reserved_payload_bytes;
        for (const auto & file : files) {
            const auto stats = file->stats();
            output.native_read_requests += stats.requests;
            output.native_system_reads += stats.system_reads;
            output.native_read_bytes += stats.bytes;
        }
    }
};

// Only immutable memory and startup counters survive construction. No file/callback fallback exists.
struct preloaded_reader final : stage_expert_reader {
    std::shared_ptr<glm_stage::memory_experts> memory;
    glm_stage::file_read_stats startup{};
    llama_stage_preload_report snapshot{};
    preloaded_reader(std::vector<std::shared_ptr<glm_stage::expert_file>> files,
            const std::vector<glm_stage::expert_address> & catalog, const llama_stage_storage_params & params, uint64_t page_budget) {
        const std::set<glm_stage::expert_address> unique(catalog.begin(), catalog.end());
        if (unique.size() != catalog.size() || catalog.size() > params.host_cache_entries) {
            throw std::invalid_argument("complete locked expert catalog exceeds entry budget or contains duplicates");
        }
        uint64_t payload = 0, pages = 0, end = 0;
        uint32_t previous_file = UINT32_MAX;
        for (const auto & address : unique) {
            if (address.file >= files.size() || address.bytes == 0 || address.bytes > glm_stage::max_expert_read_bytes ||
                    address.offset > files[address.file]->size() || address.bytes > files[address.file]->size() - address.offset ||
                    (address.file == previous_file && address.offset < end) || address.bytes > params.host_cache_bytes - payload) {
                throw std::invalid_argument("locked expert catalog is overlapping, out of bounds or exceeds payload capacity");
            }
            const uint64_t rounded = glm_stage::expert_buffer::required_locked_bytes(size_t(address.bytes));
            if (rounded > page_budget - pages) { throw std::invalid_argument("complete expert catalog exceeds native page-lock budget"); }
            payload += address.bytes; pages += rounded;
            previous_file = address.file; end = address.offset + address.bytes;
        }
        memory = std::make_shared<glm_stage::memory_experts>(params.host_cache_bytes, params.host_cache_entries, page_budget);
        for (const auto & address : catalog) {
            const auto file = files[address.file];
            memory->preload_with(address, [file,address](uint8_t * output, size_t bytes) { file->read_exact(address.offset, output, bytes); });
        }
        memory->seal(catalog);
        memory->verify_payload_locks();
        for (const auto & file : files) {
            const auto stats = file->stats();
            startup.requests += stats.requests; startup.system_reads += stats.system_reads; startup.bytes += stats.bytes;
        }
        snapshot.version = 1; snapshot.expert_slabs = uint32_t(catalog.size());
        snapshot.payload_bytes = memory->payload_bytes(); snapshot.locked_bytes = memory->locked_bytes();
        snapshot.startup_read_requests = startup.requests; snapshot.startup_read_bytes = startup.bytes;
    }
    glm_stage::expert_lease acquire(glm_stage::expert_address address) override { return memory->acquire(address); }
    void report(llama_stage_storage_report & output) const override {
        output.host_reserved_bytes = memory->payload_bytes();
        output.native_read_requests = startup.requests; output.native_system_reads = startup.system_reads; output.native_read_bytes = startup.bytes;
    }
    bool report_preload(llama_stage_preload_report & output) const override { output = snapshot; return true; }
};

struct stage_weight_source;

struct stage_context_experts final : llama_stage_context_experts {
    struct layer_context {
        stage_context_experts * owner;
        bool cpu;
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
    void report(llama_stage_bank_report & output) const override {
        const auto add = [](uint64_t & total, uint64_t value) {
            if (value > std::numeric_limits<uint64_t>::max() - total) { throw std::overflow_error("stage bank counter overflow"); }
            total += value;
        };
        for (const auto & entry : layers) {
            const auto & layer = *entry.second;
            if (layer.cpu) { ++output.cpu_banks; } else { ++output.device_banks; }
            const auto stats = layer.bank->stats();
            add(output.buffer_bytes, layer.bank->buffer_bytes());
            add(output.materializations, stats.materializations);
            add(output.slot_hits, stats.slot_hits); add(output.slot_misses, stats.slot_misses);
            add(output.copied_slabs, stats.copied_slabs); add(output.copied_bytes, stats.copied_bytes);
            add(output.failures, stats.failures);
        }
    }
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
    std::shared_ptr<stage_expert_reader> reader;
    uint64_t page_budget;
    std::unique_ptr<glm_stage::host_buffer_pool> host_pool;
    bool finalized = false;

    stage_weight_source(const llama_model & model, const llama_stage_storage_params & params, uint64_t page_budget = 0, bool lock_tensors = false) :
        model(model), params(params), page_budget(page_budget) {
        if (!model.stage_enabled || (page_budget ? model.stage.rank != 1 : model.stage.rank == 1) || model.arch != LLM_ARCH_GLM5_NEXT ||
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
        if (lock_tensors) {
            host_pool = std::make_unique<glm_stage::host_buffer_pool>(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), page_budget);
        }
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
            std::vector<std::shared_ptr<glm_stage::expert_file>> files;
            for (const auto & path : paths) {
                if (path.empty()) { throw std::invalid_argument("streamed GGUF shard path is absent"); }
                files.push_back(std::make_shared<glm_stage::expert_file>(std::filesystem::u8path(path)));
            }
            if (host_pool) {
                uint64_t pages = 0;
                for (const auto & address : catalog) {
                    const auto rounded = glm_stage::expert_buffer::required_locked_bytes(size_t(address.bytes));
                    if (rounded > page_budget - pages) { throw std::invalid_argument("expert pages exceed shared host capacity"); }
                    pages += rounded;
                }
                const auto owner = host_pool->preload_external_pages(pages, [&] {
                    return std::make_shared<preloaded_reader>(std::move(files), catalog, params, page_budget);
                }, +[](const void * pointer) -> uint64_t {
                    const auto & preload = *static_cast<const preloaded_reader *>(pointer);
                    preload.memory->verify_payload_locks(); return preload.memory->locked_bytes();
                });
                reader = std::static_pointer_cast<preloaded_reader>(owner);
            }
            else if (page_budget) { reader = std::make_shared<preloaded_reader>(std::move(files), catalog, params, page_budget); }
            else { reader = std::make_shared<desktop_reader>(model.stage.rank, std::move(files), catalog, params); }
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
        if (reader) { reader->report(output); }
    }
    bool report_preload(llama_stage_preload_report & output) const override {
        if (!page_budget) { return false; }
        return reader ? reader->report_preload(output) : true;
    }
    ggml_backend_buffer_type_t host_buffer_type(ggml_backend_buffer_type_t original) const override {
        return host_pool && ggml_backend_buft_is_host(original) ? host_pool->buffer_type() : original;
    }
    bool report_host(llama_stage_host_report & output) const override {
        if (!host_pool || !finalized) { return false; }
        host_pool->verify_all();
        const auto stats = host_pool->stats();
        output.native_payload_locks_verified = 1;
        output.page_budget = page_budget;
        output.reserved_page_bytes = stats.reserved_page_bytes; output.peak_reserved_page_bytes = stats.peak_reserved_page_bytes;
        output.expert_page_bytes = stats.external_page_bytes; output.tensor_requested_bytes = stats.requested_bytes;
        output.tensor_buffers = stats.reserved_buffers; output.pending_buffers = stats.pending_buffers;
        output.allocations = stats.allocations; output.refusals = stats.refusals; output.failures = stats.failures;
        output.last_native_error = stats.last_native_error;
        return true;
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
        layer->cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        const auto reader = this->source->reader;
        const auto & specs = registered.second.specs;
        layer->bank = std::make_unique<glm_stage::expert_bank>(backend, this->source->params.bank_slots,
            std::vector<glm_stage::expert_projection>(specs.begin(), specs.end()), std::min<uint64_t>(remaining, 1ull << 30),
            [reader](glm_stage::expert_address address) { return reader->acquire(address); },
            this->source->host_buffer_type(ggml_backend_get_default_buffer_type(backend)));
        const uint64_t bytes = layer->bank->buffer_bytes();
        remaining -= bytes;
        buffers[layer->bank->buffer_type()] += bytes;
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

std::shared_ptr<llama_stage_weight_source> llama_stage_make_preloaded_source(
        const llama_model & model, const llama_stage_preload_params & params) {
    const llama_stage_storage_params banks{1, params.bank_slots, params.host_payload_bytes, params.context_bank_bytes, params.host_entries, 0};
    return std::make_shared<stage_weight_source>(model, banks, params.host_page_bytes, params.version == 2);
}
#else
std::shared_ptr<llama_stage_weight_source> llama_stage_make_weight_source(
        const llama_model &, const llama_stage_storage_params &) {
    throw std::runtime_error("streamed stages require a build with LLAMA_GLM_RPC enabled");
}
std::shared_ptr<llama_stage_weight_source> llama_stage_make_preloaded_source(const llama_model &, const llama_stage_preload_params &) {
    throw std::runtime_error("preloaded expert stages require a build with LLAMA_GLM_RPC enabled");
}
#endif
