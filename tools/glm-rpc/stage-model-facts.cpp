#include "stage-model-facts.h"
#include "target-model.h"
#include "stage-storage.h"
#include "llama-stage.h"
#include "ggml-cpp.h"

#include <cstring>
#include <set>

namespace glm_rpc {

struct model_header_sources { std::array<std::unique_ptr<glm_stage::expert_file>,4> files; };

namespace {
void require(bool value, const char * reason) { if (!value) { throw std::invalid_argument(reason); } }
std::string hex(const digest & value) {
    std::string result;
    for (auto byte : value) { result += "0123456789abcdef"[byte>>4]; result += "0123456789abcdef"[byte&15]; }
    return result;
}

uint32_t split_value(const gguf_context * context, const char * key) {
    const auto index = gguf_find_key(context,key);
    require(index >= 0,"missing model shard field");
    const auto type = gguf_get_kv_type(context,index);
    if (type == GGUF_TYPE_UINT16) { return gguf_get_val_u16(context,index); }
    if (type == GGUF_TYPE_INT32 && gguf_get_val_i32(context,index) >= 0) { return uint32_t(gguf_get_val_i32(context,index)); }
    throw std::invalid_argument("invalid model shard field type");
}

model_header_facts inspect(const deployment_config & config, const std::array<std::filesystem::path,4> & paths, bool full) {
    config.bounds.validate(); require(config.intent_digest != digest{},"model inspection requires parsed deployment intent");
    model_header_facts result;
    result.full_file_sizes_checked = full;
    auto sources = std::make_shared<model_header_sources>();
    auto & identity = result.identity;
    identity.source_revision = compiled_source_revision(); identity.model_repository = target_model::repository;
    identity.model_revision = target_model::revision; identity.quant = target_model::quant;
    identity.tensors = target_model::tensors; identity.payload_bytes = target_model::payload_bytes;
    identity.layer_ends = config.layer_ends; identity.bounds = config.bounds;
    gguf_context_ptr schema(gguf_init_empty());
    ggml_context_ptr tensors(ggml_init({size_t(target_model::tensors+1)*ggml_tensor_overhead(),nullptr,true}));
    require(bool(schema) && bool(tensors),"cannot allocate model descriptor inventory");
    std::set<std::string> names; uint64_t payload_bytes = 0;
    for (size_t i = 0; i < paths.size(); ++i) {
        const auto & pinned = target_model::shards[i];
        sources->files[i] = std::make_unique<glm_stage::expert_file>(paths[i]);
        const auto & file = *sources->files[i];
        require(file.size() == (full ? pinned.file_bytes : pinned.header_bytes),"model source size differs from the pinned source kind");
        std::vector<uint8_t> header(size_t(pinned.header_bytes));
        file.read_exact(0,header.data(),header.size());
        const auto fingerprint = sha256({{header.data(),header.size()}});
        require(hex(fingerprint) == pinned.header_sha256,"model header differs from the exact pinned publisher bytes");
        gguf_context_ptr metadata(gguf_init_from_buffer(header.data(),header.size(),{true,nullptr}));
        require(bool(metadata) && split_value(metadata.get(),"split.no") == i &&
                split_value(metadata.get(),"split.count") == 4 && split_value(metadata.get(),"split.tensors.count") == target_model::tensors,
                "invalid or inconsistent pinned model shard descriptors");
        if (i == 0) { gguf_set_kv(schema.get(),metadata.get()); }
        for (int64_t t = 0; t < gguf_get_n_tensors(metadata.get()); ++t) {
            const std::string name = gguf_get_tensor_name(metadata.get(),t);
            require(name.size() < GGML_MAX_NAME && names.insert(name).second && names.size() <= target_model::tensors,
                    "invalid or duplicate model tensor name");
            const uint64_t offset = gguf_get_tensor_offset(metadata.get(),t);
            const uint64_t base = gguf_get_data_offset(metadata.get());
            const uint64_t bytes = gguf_get_tensor_size(metadata.get(),t);
            require(base <= pinned.file_bytes && offset <= pinned.file_bytes-base && bytes <= pinned.file_bytes-base-offset &&
                    payload_bytes <= target_model::payload_bytes && bytes <= target_model::payload_bytes-payload_bytes,
                    "model descriptor exceeds pinned file/payload bounds");
            payload_bytes += bytes;
            auto * tensor = ggml_new_tensor(tensors.get(),gguf_get_tensor_type(metadata.get(),t),GGML_MAX_DIMS,
                gguf_get_tensor_ne(metadata.get(),t));
            ggml_set_name(tensor,name.c_str()); gguf_add_tensor(schema.get(),tensor);
        }
        require(file.stats().bytes == pinned.header_bytes,"model inspection read beyond its bounded metadata range");
        result.metadata_bytes_read += file.stats().bytes;
        identity.shards.push_back({uint32_t(i),pinned.name,pinned.file_bytes,pinned.header_bytes,fingerprint});
    }
    require(names.size() == target_model::tensors && payload_bytes == target_model::payload_bytes,"incomplete pinned model descriptor inventory");
    llama_backend_init();
    llama_stage_schema_report report{1,0,0,0,0};
    require(llama_model_validate_stage_schema(schema.get(),&report) && report.tensors == target_model::tensors &&
            report.target_layers == 45 && report.nextn_layers == 1 && report.payload_bytes == target_model::payload_bytes,
            "native model schema differs from the pinned complete target");
    result.model_digest = identity_digest(identity); result.sources = std::move(sources); return result;
}
} // namespace

model_header_facts inspect_model_headers(const deployment_config & config) { return inspect(config,config.model_files,true); }
model_header_facts audit_reference_headers(const deployment_config & config, const std::array<std::filesystem::path,4> & paths) {
    return inspect(config,paths,false);
}

} // namespace glm_rpc
