#include "stage-deployment.h"
#include "stage-identity.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <iostream>

static int run(const std::vector<std::string> & args) {
    if (args.size() == 2 && args[1] == "--version") {
        std::cout << "LlamaGLM deployment schema 1, source " << glm_rpc::compiled_source_revision() << '\n'; return 0;
    }
    if (args.size() == 2 && args[1] == "--help") {
        std::cout << "Usage: llama-glm-config --check deployment.json\nValidate local deployment syntax; runtime admission is separate.\n"; return 0;
    }
    if (args.size() != 3 || args[1] != "--check") {
        std::fprintf(stderr,"Usage: llama-glm-config --check deployment.json\n"); return 2;
    }
    try {
        const auto config = glm_rpc::load_deployment(std::filesystem::u8path(args[2]));
        std::string intent;
        for (const auto byte : config.intent_digest) { intent += "0123456789abcdef"[byte>>4]; intent += "0123456789abcdef"[byte&15]; }
        nlohmann::ordered_json nodes = nlohmann::ordered_json::array();
        for (const auto & node : config.nodes) {
            nodes.push_back({{"rank",node.rank},{"address",node.address},{"data_port",node.data_port},
                {"control_port",node.control_port},{"api_port",node.api_port}});
        }
        const nlohmann::ordered_json report{{"schema_version",1},{"configuration","VALID"},{"admission","NOT_VALIDATED"},
            {"source_revision",glm_rpc::compiled_source_revision()},{"rank",config.rank},{"context_tokens",config.bounds.context},
            {"intent_sha256",intent},
            {"row_limit",config.bounds.rows},{"frame_bytes",config.bounds.frame_bytes},{"layer_ends",config.layer_ends},
            {"local_model_files",config.model_files.size()},{"tensor_schema","NOT_RUN"},{"model_payload_bytes_read",0},
            {"credentials_read",false},{"network_operations",0},{"nodes",nodes}};
        std::cout << report.dump(2) << '\n'; return 0;
    } catch (const std::invalid_argument & error) {
        std::fprintf(stderr,"GLM deployment refused: %s\n",error.what()); return 1;
    } catch (...) { std::fprintf(stderr,"GLM deployment refused: local configuration failure\n"); return 1; }
}

#ifdef _WIN32
int wmain(int argc, wchar_t ** argv) {
    try {
        std::vector<std::string> args;
        for (int i = 0; i < argc; ++i) { args.push_back(std::filesystem::path(argv[i]).u8string()); }
        return run(args);
    } catch (...) { std::fprintf(stderr,"GLM deployment refused: invalid command-line encoding\n"); return 1; }
}
#else
int main(int argc, char ** argv) { return run(std::vector<std::string>(argv,argv+argc)); }
#endif
