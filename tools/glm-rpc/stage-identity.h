#pragma once

#include "stage-wire.h"

#include <string>

namespace glm_rpc {

struct shard_identity {
    uint32_t number = 0;
    std::string name; // basename only, never a host path
    uint64_t file_bytes = 0;
    uint64_t header_bytes = 0;
    digest header_sha256{}; // exact bounded GGUF header bytes, including tokenizer/template metadata
};

// Local checked facts, not a remote deserialization or resource-admission interface.
struct execution_identity {
    std::string source_revision; // full lowercase Git SHA; not binary attestation
    std::string model_repository;
    std::string model_revision;
    std::string quant;
    std::vector<shard_identity> shards;
    uint32_t tensor_schema = 1;
    uint32_t state_layout = 1;
    uint32_t tensors = 0;
    uint64_t payload_bytes = 0;
    uint32_t target_layers = 45;
    uint32_t nextn_layers = 1; // inventoried, not executed
    uint32_t experts = 288;
    uint32_t experts_used = 8;
    uint32_t shared_experts = 1;
    std::array<uint32_t, 3> layer_ends{22, 27, 45};
    limits bounds;
};

// Binds protocol/schema/state, complete split, FP32 four-stream wire, FP16 caches, FP32 KDA,
// append-only seq-0 mode, and disabled MTP/vision/prefix/snapshots. At most 32 KiB canonical bytes.
// Callers must first validate the complete native tensor schema and check local file facts.
std::vector<uint8_t> canonical_identity(const execution_identity & value);
digest identity_digest(const execution_identity & value);
const char * compiled_source_revision();

} // namespace glm_rpc
