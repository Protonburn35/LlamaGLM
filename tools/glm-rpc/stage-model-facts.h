#pragma once

#include "stage-deployment.h"
#include "stage-identity.h"

namespace glm_rpc {

struct model_header_sources;
struct model_header_facts {
    execution_identity identity;
    digest model_digest{};
    uint64_t metadata_bytes_read = 0;
    bool full_file_sizes_checked = false;
    // Keeps native immutable source handles alive. No tensor payload access is exposed here.
    std::shared_ptr<const model_header_sources> sources;
};

// Exact pinned local full-file sizes and bounded header bytes. Still no payload/residency admission.
model_header_facts inspect_model_headers(const deployment_config & config);
// Explicit metadata-reference audit. Published sizes enter identity but are not local file facts.
model_header_facts audit_reference_headers(const deployment_config & config, const std::array<std::filesystem::path,4> & headers);

} // namespace glm_rpc
