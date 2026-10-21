#include "stage-deployment.h"
#include "stage-identity.h"
#include "stage-model-facts.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <iostream>

static int run(const std::vector<std::string> & args) {
    llama_log_set([](ggml_log_level, const char *, void *) {},nullptr);
    if (args.size() == 2 && args[1] == "--version") {
        std::cout << "LlamaGLM deployment schema 1, source " << glm_rpc::compiled_source_revision() << '\n'; return 0;
    }
    if (args.size() == 2 && args[1] == "--help") {
        std::cout << "Usage: llama-glm-config --check|--inspect deployment.json\n"
                     "       llama-glm-config --audit-headers deployment.json header1 header2 header3 header4\n"
                     "Validate syntax or exact bounded model headers; runtime admission is separate.\n"; return 0;
    }
    const bool audit = args.size() == 7 && args[1] == "--audit-headers";
    const bool inspect = args.size() == 3 && args[1] == "--inspect";
    if (!audit && !inspect && (args.size() != 3 || args[1] != "--check")) {
        std::fprintf(stderr,"Usage: llama-glm-config --check|--inspect deployment.json; --help for reference header auditing\n"); return 2;
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
        nlohmann::ordered_json report{{"schema_version",1},{"configuration","VALID"},{"admission","NOT_VALIDATED"},
            {"source_revision",glm_rpc::compiled_source_revision()},{"rank",config.rank},{"context_tokens",config.bounds.context},
            {"intent_sha256",intent},
            {"row_limit",config.bounds.rows},{"frame_bytes",config.bounds.frame_bytes},{"layer_ends",config.layer_ends},
            {"local_model_files",config.model_files.size()},{"tensor_schema","NOT_RUN"},{"model_payload_bytes_read",0},
            {"credentials_read",false},{"network_operations",0},{"nodes",nodes}};
        if (inspect || audit) {
            glm_rpc::model_header_facts facts;
            if (audit) {
                std::array<std::filesystem::path,4> headers;
                for (size_t i = 0; i < headers.size(); ++i) { headers[i] = std::filesystem::u8path(args[i+3]); }
                facts = glm_rpc::audit_reference_headers(config,headers);
            } else { facts = glm_rpc::inspect_model_headers(config); }
            std::string model_digest;
            for (const auto byte : facts.model_digest) { model_digest += "0123456789abcdef"[byte>>4]; model_digest += "0123456789abcdef"[byte&15]; }
            report["tensor_schema"] = "VALIDATED"; report["schema_checked_tensors"] = facts.identity.tensors;
            report["schema_weight_allocation_bytes"] = 0;
            report["metadata_bytes_read"] = facts.metadata_bytes_read;
            report["full_file_sizes_checked"] = facts.full_file_sizes_checked;
            report["reference_headers_only"] = audit;
            report["model_identity_sha256"] = model_digest;
            report["tensor_payloads_verified"] = false;
            report["shards"] = nlohmann::ordered_json::array();
            for (const auto & shard : facts.identity.shards) {
                std::string fingerprint;
                for (const auto byte : shard.header_sha256) { fingerprint += "0123456789abcdef"[byte>>4]; fingerprint += "0123456789abcdef"[byte&15]; }
                report["shards"].push_back({{"number",shard.number},{"name",shard.name},{"file_bytes",shard.file_bytes},
                    {"header_bytes",shard.header_bytes},{"header_sha256",fingerprint}});
            }
        }
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
