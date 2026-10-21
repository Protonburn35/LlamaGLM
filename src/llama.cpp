#include "llama.h"

#include "llama-impl.h"
#include "llama-version.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"
#include "llama-stage-provider.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

const char * llama_load_mode_name(enum llama_load_mode load_mode) {
    switch (load_mode) {
        case LLAMA_LOAD_MODE_AUTO:
            return "auto";
        case LLAMA_LOAD_MODE_NONE:
            return "none";
        case LLAMA_LOAD_MODE_MMAP:
            return "mmap";
        case LLAMA_LOAD_MODE_MLOCK:
            return "mlock";
        case LLAMA_LOAD_MODE_MMAP_MLOCK:
            return "mmap+mlock";
        case LLAMA_LOAD_MODE_DIRECT_IO:
            return "dio";
    }
    GGML_ABORT("fatal error");
}

enum llama_load_mode llama_load_mode_from_str(const char * str) {
    if (std::strcmp(str, "auto")       == 0) { return LLAMA_LOAD_MODE_AUTO;       }
    if (std::strcmp(str, "none")       == 0) { return LLAMA_LOAD_MODE_NONE;       }
    if (std::strcmp(str, "mmap")       == 0) { return LLAMA_LOAD_MODE_MMAP;       }
    if (std::strcmp(str, "mlock")      == 0) { return LLAMA_LOAD_MODE_MLOCK;      }
    if (std::strcmp(str, "mmap+mlock") == 0) { return LLAMA_LOAD_MODE_MMAP_MLOCK; }
    if (std::strcmp(str, "dio")        == 0) { return LLAMA_LOAD_MODE_DIRECT_IO;  }
    GGML_ABORT("unknown load mode: %s", str);
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

const char * llama_version(void) {
    return LLAMA_VERSION;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// returns true on success
static bool llama_prepare_model_devices(const llama_model_params & params, llama_model * model) {
    // create list of devices to use with this model
    if (params.devices) {
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            size_t n_devs = 0;
            while (params.devices[n_devs]) {
                n_devs++;
            }
            if (n_devs == 0) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: creating a Meta device with %zu devices\n", __func__, n_devs);
            for (size_t i = 0; i < n_devs; ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s\n", __func__, i, ggml_backend_dev_name(params.devices[i]));
            }
            model->get_split_state_ud.n_devices = n_devs;
            model->get_split_state_ud.model = model;
            model->devices.push_back({
                true, ggml_backend_meta_device(
                params.devices, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
                model->devices.push_back({false, *dev});
            }
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<llama_device> gpus;
        std::vector<llama_device> igpus;
        std::vector<llama_device> rpc_servers;

        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            std::vector<ggml_backend_dev_t> devs;
            devs.reserve(ggml_backend_dev_count());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type()) {
                    LLAMA_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    continue;
                }
                devs.push_back(dev);
            }
            if (devs.empty()) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }

            LLAMA_LOG_INFO("%s: creating a Meta device for tensor parallelism from %zu devices:\n", __func__, devs.size());
            for (size_t i = 0; i < devs.size(); ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s (%s)\n", __func__, i, ggml_backend_dev_name(devs[i]), ggml_backend_dev_description(devs[i]));
            }

            GGML_ASSERT(!devs.empty());
            model->get_split_state_ud.n_devices = devs.size();
            model->get_split_state_ud.model     = model;
            gpus.push_back({
                true, ggml_backend_meta_device(
                devs.data(), devs.size(), llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                            rpc_servers.push_back({false, dev});
                        } else {
                            // check if there is already a GPU with the same device id
                            ggml_backend_dev_props props;
                            ggml_backend_dev_get_props(dev, &props);
                            auto it = std::find_if(gpus.begin(), gpus.end(), [&props](const llama_device & d) {
                                ggml_backend_dev_props d_props;
                                ggml_backend_dev_get_props(d.dev, &d_props);
                                if (props.device_id && d_props.device_id) {
                                    return strcmp(props.device_id, d_props.device_id) == 0;
                                }
                                return false;
                            });

                            if (it != gpus.end()) {
                                LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                        __func__,
                                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                        props.device_id ? props.device_id : "unknown id",
                                        ggml_backend_dev_name(it->dev), ggml_backend_dev_description(it->dev));
                            } else {
                                gpus.push_back({false, dev});
                            }
                        }
                        break;
                    }

                    case GGML_BACKEND_DEVICE_TYPE_IGPU:
                        // igpus.empty() - workaround for integrated devices seen by multiple backends
                        // ref: https://github.com/ggml-org/llama.cpp/pull/23897
                        // ggml_backend_dev_backend_reg - allow devices of the same backend regardless if integrated
                        // ref: https://github.com/ggml-org/llama.cpp/pull/23897#issuecomment-5264222997
                        if (igpus.empty() || ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(igpus.back().dev)) {
                            igpus.push_back({false, dev});
                        }
                        break;
                    case GGML_BACKEND_DEVICE_TYPE_META:
                        GGML_ABORT("fatal error");
                }
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no discrete GPUs were found
        // (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
        if (gpus.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE && !model->devices.empty()) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                return false;
            }
            llama_device main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (const auto & dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev.dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params,
        const llama_stage_model_params * stage, bool strict_metadata = false, const llama_stage_storage_params * storage = nullptr,
        const llama_stage_preload_params * preload = nullptr) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.load_mode,
            params.check_tensors, params.no_alloc, params.load_mtp, params.kv_overrides, params.tensor_buft_overrides, strict_metadata);

        ml.lazy.mode = params.lazy_mode;

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get());
        if (!ok) {
            return {-1, nullptr};
        }

        auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            GGML_ABORT("fatal error: model does not implement llama_model_base");
        }

        // loading time will be recalculated after the first eval, so
        // we take page faults deferred by mmap() into consideration
        model->t_load_us = 0;
        time_meas tm(model->t_load_us);

        model->t_start_us = tm.t_start_us;

        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (stage) {
            if (model->arch != LLM_ARCH_GLM5_NEXT || stage->layer_ends[2] != model->hparams.n_layer()) {
                throw std::invalid_argument("stage plan must cover the complete GLM5Next target graph");
            }
            model->stage = *stage;
            model->stage_enabled = true;
        }
        if (storage) { model->stage_streamed = true; model->stage_storage = *storage; }
        if (preload) { model->stage_preloaded = true; model->stage_preload = *preload; }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return {0, model_ptr.release()};
        }

        if (!model->load_tensors(ml)) {
            return {-2, nullptr};
        }

        return {0, model_ptr.release()};
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return {-1, nullptr};
    }
}

static void llama_check_stage_metadata_limits(const gguf_context * metadata) {
    const int64_t arch = gguf_find_key(metadata, "general.architecture");
    if (arch < 0 || gguf_get_kv_type(metadata, arch) != GGUF_TYPE_STRING ||
            std::strcmp(gguf_get_val_str(metadata, arch), "glm5-next") != 0 || gguf_get_n_kv(metadata) > 10000 ||
            gguf_get_n_tensors(metadata) <= 0 || gguf_get_n_tensors(metadata) > 100000) {
        throw std::invalid_argument("schema check requires bounded complete GLM5Next metadata");
    }
    const auto integer = [&](const char * suffix, uint32_t low, uint32_t high, bool required = true) {
        const std::string key = std::string("glm5-next.") + suffix;
        const int64_t id = gguf_find_key(metadata, key.c_str());
        if (id < 0 && !required) { return uint32_t(0); }
        if (id < 0 || gguf_get_kv_type(metadata, id) != GGUF_TYPE_UINT32) { throw std::invalid_argument("invalid schema geometry field: " + key); }
        const uint32_t value = gguf_get_val_u32(metadata, id);
        if (value < low || value > high) { throw std::invalid_argument("schema geometry field exceeds limits: " + key); }
        return value;
    };
    const uint32_t layers = integer("block_count", 1, 256);
    integer("nextn_predict_layers", 0, layers - 1, false);
    integer("embedding_length", 1, 65536);
    integer("attention.head_count", 1, 1024);
    integer("context_length", 1, 1048576);
    integer("hyper_connection.count", 4, 4);
    integer("hyper_connection.sinkhorn_iterations", 1, 128);
    integer("ssm.conv_kernel", 1, 64);
    integer("kda.head_dim", 1, 1024);
    integer("attention.key_length_mla", 1, 65536);
    integer("attention.value_length_mla", 1, 65536);
    integer("attention.q_lora_rank", 1, 65536);
    integer("attention.kv_lora_rank", 1, 65536);
    integer("attention.indexer.head_count", 1, 1024);
    integer("attention.indexer.key_length", 1, 1024);
    const uint32_t pool = integer("attention.indexer.kpool", 2, 64);
    if (integer("attention.indexer.top_k", 1, 1048576) % pool) { throw std::invalid_argument("schema index top-k must cover complete pools"); }
    const uint32_t experts = integer("expert_count", 1, 512);
    integer("expert_used_count", 1, experts);
    const uint32_t groups = integer("expert_group_count", 0, experts - 1, false);
    const uint32_t groups_used = integer("expert_group_used_count", 0, experts, false);
    if (groups > 1 && (experts % groups || groups_used == 0 || groups_used >= groups)) {
        throw std::invalid_argument("invalid schema expert grouping");
    }
    integer("expert_shared_count", 1, 512);
    integer("leading_dense_block_count", 0, layers, false);
    integer("vocab_size", 1, 1048576, false);
    const auto choice = [&](const char * suffix, std::initializer_list<const char *> values) {
        const std::string key = std::string("glm5-next.") + suffix;
        const int64_t id = gguf_find_key(metadata, key.c_str());
        if (id < 0) { return; }
        if (gguf_get_kv_type(metadata, id) == GGUF_TYPE_STRING) {
            for (const char * value : values) {
                if (std::strcmp(gguf_get_val_str(metadata, id), value) == 0) { return; }
            }
        }
        throw std::invalid_argument("unsupported schema geometry choice: " + key);
    };
    choice("rope.scaling.type", {"none", "linear", "yarn", "longrope"});
    choice("classifier.activation", {"gelu", "silu", "tanh"});
    const auto array_field = [&](const char * suffix, uint32_t maximum, bool scalar) {
        const std::string key = std::string("glm5-next.") + suffix;
        const int64_t id = gguf_find_key(metadata, key.c_str());
        if (id < 0) { throw std::invalid_argument("missing schema geometry field: " + key); }
        if (scalar && gguf_get_kv_type(metadata, id) == GGUF_TYPE_UINT32) {
            if (gguf_get_val_u32(metadata, id) > maximum) { throw std::invalid_argument("schema geometry scalar exceeds limit"); }
            return;
        }
        if (gguf_get_kv_type(metadata, id) != GGUF_TYPE_ARRAY || gguf_get_arr_n(metadata, id) != layers ||
                (gguf_get_arr_type(metadata, id) != GGUF_TYPE_UINT32 && gguf_get_arr_type(metadata, id) != GGUF_TYPE_INT32)) {
            throw std::invalid_argument("invalid schema geometry array: " + key);
        }
        for (uint32_t i = 0; i < layers; ++i) {
            const void * data = gguf_get_arr_data(metadata, id);
            const int64_t value = gguf_get_arr_type(metadata, id) == GGUF_TYPE_UINT32 ?
                int64_t(static_cast<const uint32_t *>(data)[i]) : int64_t(static_cast<const int32_t *>(data)[i]);
            if (value < 0 || uint64_t(value) > maximum) { throw std::invalid_argument("schema geometry array exceeds limit"); }
        }
    };
    array_field("attention.head_count_kv", 1, false);
    array_field("feed_forward_length", 1048576, true);
    array_field("expert_feed_forward_length", 1048576, true);
    const int64_t tokens = gguf_find_key(metadata, "tokenizer.ggml.tokens");
    if (tokens >= 0 && (gguf_get_kv_type(metadata, tokens) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(metadata, tokens) != GGUF_TYPE_STRING || gguf_get_arr_n(metadata, tokens) > 1048576)) {
        throw std::invalid_argument("schema vocabulary exceeds metadata limits");
    }
    uint64_t total = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
        const auto type = gguf_get_tensor_type(metadata, i);
        if (type < 0 || type >= GGML_TYPE_COUNT || ggml_blck_size(type) <= 0 || ggml_type_size(type) == 0) {
            throw std::invalid_argument("invalid schema tensor type");
        }
        const auto * ne = gguf_get_tensor_ne(metadata, i);
        uint64_t bytes = ggml_type_size(type);
        for (size_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (ne[dim] <= 0 || ne[dim] > (1 << 24) || (dim == 0 && ne[dim] % ggml_blck_size(type))) {
                throw std::invalid_argument("invalid schema tensor dimension");
            }
            const uint64_t factor = dim == 0 ? ne[dim] / ggml_blck_size(type) : ne[dim];
            if (bytes > (1ull << 40) / factor) { throw std::invalid_argument("schema tensor exceeds byte limit"); }
            bytes *= factor;
        }
        if (total > (1ull << 44) - bytes) { throw std::invalid_argument("schema tensor inventory exceeds byte limit"); }
        total += bytes;
    }
}

bool llama_model_validate_stage_schema(gguf_context * metadata, llama_stage_schema_report * report) {
    if (!metadata || !report || report->version != 1) {
        LLAMA_LOG_ERROR("%s: complete metadata and a version-1 report are required\n", __func__);
        return false;
    }
    *report = {1, 0, 0, 0, 0};
    try {
        llama_check_stage_metadata_limits(metadata);
        auto params = llama_model_default_params();
        ggml_backend_dev_t cpu_only[] = {nullptr};
        params.devices = cpu_only;
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.lazy_mode = LLAMA_LAZY_MODE_OFF;
        params.no_alloc = true;
        params.load_mtp = true;
        params.use_extra_bufts = false;
        params.progress_callback = nullptr;
        std::vector<std::string> splits;
        const auto loaded = llama_model_load(metadata, nullptr, nullptr, "", splits, nullptr, params, nullptr, true);
        std::unique_ptr<llama_model> model(loaded.second);
        if (loaded.first != 0 || !model) { return false; }
        uint64_t bytes = 0;
        for (const auto & entry : model->tensors_by_name) {
            if (entry.second->data || !entry.second->buffer || ggml_backend_buffer_get_size(entry.second->buffer) != 0) {
                throw std::runtime_error("schema inspection unexpectedly allocated weight data");
            }
            bytes += ggml_nbytes(entry.second);
        }
        *report = {1, uint32_t(model->tensors_by_name.size()), model->hparams.n_layer(), model->hparams.n_layer_nextn, bytes};
        return true;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: schema refused: %s\n", __func__, error.what());
        return false;
    }
}

static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params,
        const llama_stage_model_params * stage = nullptr,
        const llama_stage_storage_params * storage = nullptr,
        const llama_stage_preload_params * preload = nullptr) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }
    if (stage && (stage->version != 1 || stage->rank > 2 || stage->layer_ends[0] == 0 ||
            stage->layer_ends[0] >= stage->layer_ends[1] || stage->layer_ends[1] >= stage->layer_ends[2] ||
            params.load_mtp || params.vocab_only || params.lazy_mode != LLAMA_LAZY_MODE_OFF ||
            (!metadata && params.load_mode != LLAMA_LOAD_MODE_NONE))) {
        LLAMA_LOG_ERROR("%s: unsupported stage version/rank/ranges or load mode; use three nonempty target stages, buffered NONE, no lazy reads or MTP\n", __func__);
        return nullptr;
    }
    if (storage && (!stage || stage->rank == 1 || metadata || file || params.no_alloc || params.check_tensors || params.no_host ||
            storage->version != 1 || storage->bank_slots < 8 || storage->bank_slots > 512 ||
            storage->host_cache_bytes == 0 || storage->host_cache_bytes > (1ull << 40) ||
            storage->context_bank_bytes == 0 || storage->context_bank_bytes > (1ull << 36) ||
            storage->host_cache_entries == 0 || storage->host_cache_entries > 4096 ||
            storage->max_active_reads == 0 || storage->max_active_reads > 8)) {
        LLAMA_LOG_ERROR("%s: invalid desktop storage version/rank/budgets or unsupported file/check/allocation policy\n", __func__);
        return nullptr;
    }
    if (preload && (!stage || stage->rank != 1 || storage || metadata || file || params.no_alloc || params.check_tensors || params.no_host ||
            (preload->version != 1 && preload->version != 2) || preload->reserved != 0 || preload->bank_slots < 8 || preload->bank_slots > 512 ||
            (preload->version == 2 && (params.use_extra_bufts || params.tensor_buft_overrides)) ||
            preload->host_payload_bytes == 0 || preload->host_payload_bytes > (1ull << 40) ||
            preload->host_page_bytes == 0 || preload->host_page_bytes > (1ull << 40) ||
            preload->context_bank_bytes == 0 || preload->context_bank_bytes > (1ull << 36) ||
            preload->host_entries == 0 || preload->host_entries > 100000)) {
        LLAMA_LOG_ERROR("%s: invalid locked-expert preload version/rank/budgets or file/allocation policy\n", __func__);
        return nullptr;
    }
    // TODO: remove
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    const auto [status, model] = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, params, stage, false, storage, preload);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        if (model) {
            llama_model_free(model);
        }
        return nullptr;
    }

    return model;
}

struct llama_model * llama_model_init_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params) {
    GGML_ASSERT(metadata != nullptr);
    std::string path_model;
    std::vector<std::string> splits = {};
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    return llama_model_load_from_file_impl(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, /*file*/ nullptr, params);
}
// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_stage_from_file(
        const char * path_model, struct llama_model_params params, struct llama_stage_model_params stage) {
    if (!path_model) {
        LLAMA_LOG_ERROR("%s: model path cannot be null\n", __func__);
        return nullptr;
    }
    std::vector<std::string> splits;
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, nullptr, params, &stage);
}

struct llama_model * llama_model_load_stage_streamed_from_file(
        const char * path_model, llama_model_params params, llama_stage_model_params stage, llama_stage_storage_params storage) {
    if (!path_model) { LLAMA_LOG_ERROR("%s: model path cannot be null\n", __func__); return nullptr; }
    std::vector<std::string> splits;
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, nullptr, params, &stage, &storage);
}

bool llama_model_stage_storage_report(const llama_model * model, llama_stage_storage_report * report) {
    if (!report || report->version != 1) { return false; }
    *report = {};
    report->version = 1;
    if (!model || !model->stage_source) { return false; }
    try {
        model->stage_source->report(*report);
        for (const auto & entry : model->memory_breakdown()) {
            if (entry.second > UINT64_MAX - report->model_buffer_bytes) { throw std::overflow_error("model buffer report overflow"); }
            report->model_buffer_bytes += entry.second;
        }
        return true;
    } catch (...) {
        *report = {};
        report->version = 1;
        return false;
    }
}

struct llama_model * llama_model_load_stage_preloaded_from_file(
        const char * path_model, llama_model_params params, llama_stage_model_params stage, llama_stage_preload_params preload) {
    if (!path_model) { LLAMA_LOG_ERROR("%s: model path cannot be null\n", __func__); return nullptr; }
    std::vector<std::string> splits;
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, nullptr, params, &stage, nullptr, &preload);
}

bool llama_model_stage_preload_report(const llama_model * model, llama_stage_preload_report * report) {
    if (!report || report->version != 1) { return false; }
    *report = {}; report->version = 1;
    if (!model || !model->stage_preloaded || !model->stage_source) { return false; }
    try { return model->stage_source->report_preload(*report); }
    catch (...) { *report = {}; report->version = 1; return false; }
}

bool llama_model_stage_host_report(const llama_model * model, llama_stage_host_report * report) {
    if (!report || report->version != 1) { return false; }
    *report = {}; report->version = 1;
    if (!model || !model->stage_source) { return false; }
    try { return model->stage_source->report_host(*report); }
    catch (...) { *report = {}; report->version = 1; return false; }
}

struct llama_model * llama_model_init_stage_from_user(
        struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud, struct llama_model_params params, struct llama_stage_model_params stage) {
    if (!metadata || (!params.no_alloc && !set_tensor_data)) {
        LLAMA_LOG_ERROR("%s: metadata and a tensor provider are required\n", __func__);
        return nullptr;
    }
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    std::vector<std::string> splits;
    return llama_model_load_from_file_impl(metadata, set_tensor_data, set_tensor_data_ud, "", splits, nullptr, params, &stage);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, splits.front(), splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_file_ptr(FILE * file, struct llama_model_params params) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }
    std::string path_model;
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, file, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    if (model && model->stage_enabled) {
        LLAMA_LOG_ERROR("%s: partial stage models cannot be saved as complete GGUF models\n", __func__);
        return;
    }
    llama_model_saver ms(model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

