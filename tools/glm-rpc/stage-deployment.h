#pragma once

#include "stage-net.h"

#include <filesystem>

namespace glm_rpc {

struct deployment_node {
    uint32_t rank = 0;
    std::string address;
    uint16_t data_port = 0, control_port = 0, api_port = 0;
};

// Local configuration only. Parsing does not open model/credential files, probe resources,
// contact peers or grant admission. Actual headers and all allocations still need validation.
struct deployment_config {
    digest intent_digest{}; // Parser-produced common intent; keep these facts immutable after parsing.
    uint32_t rank = 0;
    std::array<deployment_node,3> nodes;
    std::array<std::filesystem::path,4> model_files;
    std::filesystem::path secret_file;
    std::array<uint32_t,3> layer_ends{22,27,45};
    limits bounds;
};

constexpr size_t max_deployment_bytes = 64u<<10;
deployment_config parse_deployment(const std::string & text, const std::filesystem::path & local_directory);
deployment_config load_deployment(const std::filesystem::path & path);

} // namespace glm_rpc
