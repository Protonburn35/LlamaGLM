#include "stage-identity.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace glm_rpc {
namespace {

bool nonzero(const digest & value) {
    return std::any_of(value.begin(), value.end(), [](uint8_t byte) { return byte != 0; });
}

void revision(const std::string & value) {
    if (value.size() != 40 || !std::all_of(value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); })) {
        throw std::invalid_argument("identity requires full lowercase source/model revisions");
    }
}

void text_field(const std::string & value, bool basename) {
    if (value.empty() || value.size() > 255 || value == "." || value == "..") {
        throw std::invalid_argument("identity text exceeds limits");
    }
    for (const char c : value) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                c == '_' || c == '-' || c == '.' || (!basename && c == '/'))) {
            throw std::invalid_argument("identity text contains unsupported characters");
        }
    }
}

struct encoding {
    std::vector<uint8_t> data;
    void integer(uint64_t value, size_t bytes) {
        for (size_t i = 0; i < bytes; ++i) { data.push_back(uint8_t(value >> (8 * i))); }
    }
    void string(const std::string & value) {
        integer(value.size(), 4);
        data.insert(data.end(), value.begin(), value.end());
    }
};

} // namespace

std::vector<uint8_t> canonical_identity(const execution_identity & value) {
    revision(value.source_revision);
    revision(value.model_revision);
    text_field(value.model_repository, false);
    text_field(value.quant, true);
    value.bounds.validate();
    if (value.tensor_schema != 1 || value.state_layout != 1 || value.tensors == 0 || value.tensors > 100000 ||
            value.payload_bytes == 0 || value.payload_bytes > (1ull << 44) || value.target_layers < 3 || value.target_layers > 256 ||
            value.nextn_layers > 1 || value.experts == 0 || value.experts > 512 || value.experts_used == 0 ||
            value.experts_used > value.experts || value.shared_experts == 0 || value.shared_experts > 512 ||
            value.layer_ends[0] == 0 || value.layer_ends[0] >= value.layer_ends[1] ||
            value.layer_ends[1] >= value.layer_ends[2] || value.layer_ends[2] != value.target_layers ||
            value.shards.empty() || value.shards.size() > 100) {
        throw std::invalid_argument("unsupported or incomplete execution identity");
    }
    std::vector<const shard_identity *> shards;
    for (const auto & shard : value.shards) { shards.push_back(&shard); }
    std::sort(shards.begin(), shards.end(), [](const shard_identity * a, const shard_identity * b) { return a->number < b->number; });
    std::set<std::string> names;
    uint64_t capacity = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        const auto & shard = *shards[i];
        text_field(shard.name, true);
        std::string folded = shard.name;
        for (char & c : folded) { if (c >= 'A' && c <= 'Z') { c = char(c - 'A' + 'a'); } }
        if (shard.name.size() < 6 || shard.name.compare(shard.name.size() - 5, 5, ".gguf") != 0 ||
                shard.number != i || !names.insert(folded).second || !nonzero(shard.header_sha256) ||
                shard.header_bytes < 24 || shard.header_bytes > (32ull << 20) || shard.file_bytes < shard.header_bytes ||
                shard.file_bytes > (1ull << 44)) {
            throw std::invalid_argument("invalid or incomplete shard identity");
        }
        // At most 100 shards * 16 TiB each, within uint64_t.
        capacity += shard.file_bytes - shard.header_bytes;
    }
    if (capacity < value.payload_bytes) { throw std::invalid_argument("shard sizes cannot hold the declared tensor payloads"); }

    encoding out;
    const uint8_t domain[] = "LlamaGLM execution identity v1";
    out.data.insert(out.data.end(), domain, domain + sizeof(domain));
    out.integer(1, 4); // identity schema
    out.integer(1, 2); out.integer(0, 2); // wire major/minor
    out.string(value.source_revision);
    out.string("glm5-next");
    out.string(value.model_repository);
    out.string(value.model_revision);
    out.string(value.quant);
    out.integer(value.tensor_schema, 4); out.integer(value.state_layout, 4);
    out.integer(value.tensors, 4); out.integer(value.payload_bytes, 8);
    out.integer(value.target_layers, 4); out.integer(value.nextn_layers, 4);
    out.integer(value.experts, 4); out.integer(value.experts_used, 4); out.integer(value.shared_experts, 4);
    for (const uint32_t end : value.layer_ends) { out.integer(end, 4); }
    out.integer(value.bounds.hidden, 4); out.integer(value.bounds.streams, 4);
    out.integer(value.bounds.rows, 4); out.integer(value.bounds.context, 4); out.integer(value.bounds.frame_bytes, 4);
    out.integer(max_control_bytes, 4); out.integer(max_chunk_bytes, 8);
    out.string("f32le:hidden,stream,row"); out.string("f16:mla,index"); out.string("f32:kda");
    out.string("causal:seq0:append-only");
    out.integer(0, 4); // MTP, vision, prefix reuse and snapshots unsupported
    out.integer(shards.size(), 4);
    for (const auto * item : shards) {
        const auto & shard = *item;
        out.integer(shard.number, 4); out.string(shard.name);
        out.integer(shard.file_bytes, 8); out.integer(shard.header_bytes, 8);
        out.data.insert(out.data.end(), shard.header_sha256.begin(), shard.header_sha256.end());
    }
    if (out.data.size() > (32u << 10)) { throw std::invalid_argument("canonical identity exceeds limit"); }
    return out.data;
}

digest identity_digest(const execution_identity & value) {
    const auto encoded = canonical_identity(value);
    return sha256({{encoded.data(), encoded.size()}});
}

const char * compiled_source_revision() {
    return GLM_RPC_SOURCE_REVISION;
}

} // namespace glm_rpc
