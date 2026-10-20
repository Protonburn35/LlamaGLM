#include "stage-plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace {

constexpr uint64_t header_limit = 32ULL << 20;

struct header_reader {
    FILE * file;
    uint64_t position = 0;
};

size_t read_header(void * userdata, void * output, uint64_t offset, size_t size) {
    auto & reader = *static_cast<header_reader *>(userdata);
    if (offset > header_limit || size > header_limit - offset) {
        return 0;
    }
    if (reader.position != offset) {
#ifdef _WIN32
        if (_fseeki64(reader.file, int64_t(offset), SEEK_SET) != 0) {
#else
        if (fseeko(reader.file, off_t(offset), SEEK_SET) != 0) {
#endif
            return 0;
        }
    }
    const size_t count = std::fread(output, 1, size, reader.file);
    reader.position = offset + count;
    return count;
}

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

struct close_file {
    void operator()(FILE * file) const { std::fclose(file); }
};

gguf_ptr load_header(const std::string & path) {
    std::unique_ptr<FILE, close_file> file(std::fopen(path.c_str(), "rb"));
    if (!file) {
        throw std::runtime_error("cannot open GGUF header");
    }
    header_reader reader{file.get()};
    gguf_ptr ctx(gguf_init_from_callback(read_header, &reader, 65536, header_limit, {true, nullptr}), &gguf_free);
    if (!ctx) {
        throw std::runtime_error("invalid, truncated or oversized GGUF metadata");
    }
    return ctx;
}

uint32_t split_value(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id >= 0) {
        const auto type = gguf_get_kv_type(ctx, id);
        if (type == GGUF_TYPE_UINT16) {
            return gguf_get_val_u16(ctx, id);
        }
        if (type == GGUF_TYPE_INT32 && gguf_get_val_i32(ctx, id) >= 0) {
            return uint32_t(gguf_get_val_i32(ctx, id));
        }
    }
    throw std::invalid_argument(std::string("missing or incorrectly typed split field: ") + key);
}

uint32_t context_value(const std::string & text) {
    uint32_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument("invalid context integer");
    }
    return result;
}

const char * role_name(glm_stage::tensor_role role) {
    switch (role) {
        case glm_stage::tensor_role::trunk:  return "trunk";
        case glm_stage::tensor_role::input:  return "input";
        case glm_stage::tensor_role::output: return "output";
        case glm_stage::tensor_role::draft:  return "draft";
    }
    throw std::logic_error("invalid tensor role");
}

json make_manifest(const std::vector<std::string> & paths, const glm_stage::plan & split) {
    if (paths.empty() || paths.size() > 100) {
        throw std::invalid_argument("provide all GGUF shards, with metadata shard first (at most 100)");
    }
    auto first = load_header(paths.front());
    const auto model = glm_stage::geometry::from_gguf(first.get());
    split.validate(model.layers);
    const auto state = glm_stage::state_budgets(split, model);
    const uint32_t shards = split_value(first.get(), "split.count");
    const uint32_t tensor_count = split_value(first.get(), "split.tensors.count");
    if (shards != paths.size() || split_value(first.get(), "split.no") != 0 || tensor_count > 100000) {
        throw std::invalid_argument("incomplete or excessive shard/tensor inventory");
    }
    json result = {
        {"schema_version", glm_stage::plan_version}, {"architecture", "glm5-next"},
        {"context_tokens", split.context}, {"cache_dtype", "f16"}, {"kda_state_dtype", "f32"},
        {"activation_dtype", "f32"}, {"boundary_bytes_per_token", model.boundary_bytes(1)},
        {"admission", "NOT_VALIDATED"}, {"tensor_payloads_verified", false},
        {"limitations", {"Metadata inventory only; no model tensor payload is read.",
            "State bytes exclude allocator overhead, cache bookkeeping, scratch, graphs, OS, display and queues.",
            "RAM/VRAM placement, Windows residency and runtime execution are not validated.",
            "NextN weights are inventoried separately; MTP execution remains disabled."}},
        {"shards", json::array()}, {"stages", json::array()}, {"tensors", json::array()}
    };
    for (size_t rank = 0; rank < 3; ++rank) {
        result["stages"].push_back({
            {"rank", rank}, {"layer_begin", split.stages[rank].begin}, {"layer_end", split.stages[rank].end},
            {"model_weights_bytes", 0ULL}, {"routed_expert_bytes", 0ULL}, {"draft_weights_bytes", 0ULL},
            {"kda_layers", state[rank].kda_layers}, {"dsa_layers", state[rank].dsa_layers},
            {"mla_cache_bytes", state[rank].mla_bytes}, {"index_cache_bytes", state[rank].index_bytes},
            {"kda_state_bytes", state[rank].recurrent_bytes}, {"target_state_bytes", state[rank].total()},
            {"inference_ssd_allowed", rank != 1}
        });
    }
    std::set<uint32_t> seen_shards;
    std::set<std::string> seen_tensors;
    std::set<uint32_t> seen_layers;
    for (size_t index = 0; index < paths.size(); ++index) {
        auto ctx = index == 0 ? std::move(first) : load_header(paths[index]);
        const uint32_t number = split_value(ctx.get(), "split.no");
        if (split_value(ctx.get(), "split.count") != shards || number >= shards ||
                !seen_shards.insert(number).second || split_value(ctx.get(), "split.tensors.count") != tensor_count) {
            throw std::invalid_argument("inconsistent or duplicate GGUF shards");
        }
        result["shards"].push_back({{"number", number}, {"metadata_bytes", gguf_get_data_offset(ctx.get())}});
        for (int64_t t = 0; t < gguf_get_n_tensors(ctx.get()); ++t) {
            const std::string name = gguf_get_tensor_name(ctx.get(), t);
            if (!seen_tensors.insert(name).second) {
                throw std::invalid_argument("duplicate tensor across shards: " + name);
            }
            const auto owner = glm_stage::assign_tensor(split, model, name);
            const auto * shape = gguf_get_tensor_ne(ctx.get(), t);
            const uint64_t bytes = gguf_get_tensor_size(ctx.get(), t);
            const uint64_t offset = glm_stage::checked_add(gguf_get_data_offset(ctx.get()), gguf_get_tensor_offset(ctx.get(), t));
            glm_stage::checked_add(offset, bytes);
            if (owner.routed_experts && shape[2] != model.experts) {
                throw std::invalid_argument("expert tensor does not contain every routed expert: " + name);
            }
            if (owner.layer >= 0 && owner.role == glm_stage::tensor_role::trunk) {
                seen_layers.insert(uint32_t(owner.layer));
            }
            auto & stage = result["stages"][owner.rank];
            const char * category = owner.role == glm_stage::tensor_role::draft ? "draft_weights_bytes" : "model_weights_bytes";
            stage[category] = glm_stage::checked_add(stage[category].get<uint64_t>(), bytes);
            if (owner.routed_experts && owner.role == glm_stage::tensor_role::trunk) {
                stage["routed_expert_bytes"] = glm_stage::checked_add(stage["routed_expert_bytes"].get<uint64_t>(), bytes);
            }
            result["tensors"].push_back({{"name", name}, {"shard", number}, {"offset", offset}, {"bytes", bytes},
                {"shape", {shape[0], shape[1], shape[2], shape[3]}},
                {"type", ggml_type_name(gguf_get_tensor_type(ctx.get(), t))}, {"rank", owner.rank},
                {"layer", owner.layer}, {"role", role_name(owner.role)}, {"routed_experts", owner.routed_experts}});
        }
    }
    if (seen_tensors.size() != tensor_count || seen_layers.size() != model.layers ||
            !seen_tensors.count("token_embd.weight") || !seen_tensors.count("output.weight") ||
            !seen_tensors.count("output_norm.weight")) {
        throw std::invalid_argument("incomplete model tensor inventory");
    }
    for (auto & stage : result["stages"]) {
        stage["all_weights_resident_lower_bound_bytes"] = glm_stage::checked_add(stage["model_weights_bytes"].get<uint64_t>(),
                                                                  stage["target_state_bytes"].get<uint64_t>());
    }
    result["draft_state_bytes_if_enabled"] = glm_stage::checked_mul(split.context, (model.kv_latent + 3ULL * model.index_head) * 2);
    result["tensor_count"] = seen_tensors.size();
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string counts = "22,5,18";
        uint32_t context = glm_stage::max_context;
        std::string output;
        std::vector<std::string> paths;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: llama-glm-plan [--split 22,5,18] [--context 1048576] [--output manifest.json] shard1.gguf ...\n"
                          << "Reads at most 32 MiB of metadata per shard, without loading tensor payloads.\n";
                return 0;
            }
            if (arg == "--version") {
                std::cout << "LlamaGLM stage plan schema " << glm_stage::plan_version << '\n';
                return 0;
            }
            if (arg == "--split" || arg == "--context" || arg == "--output") {
                if (++i == argc) {
                    throw std::invalid_argument("missing option value");
                }
                if (arg == "--split") { counts = argv[i]; }
                else if (arg == "--context") { context = context_value(argv[i]); }
                else { output = argv[i]; }
            } else if (arg.compare(0, 2, "--") == 0) {
                throw std::invalid_argument("unknown option");
            } else {
                paths.push_back(arg);
            }
        }
        const auto manifest = make_manifest(paths, glm_stage::plan::from_counts(counts, context));
        if (output.empty()) {
            std::cout << manifest.dump(2) << '\n';
        } else {
            std::ofstream stream(output);
            stream << manifest.dump(2) << '\n';
            if (!stream) {
                throw std::runtime_error("cannot write manifest");
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "LlamaGLM plan refused: " << error.what() << '\n';
        return 1;
    }
}
