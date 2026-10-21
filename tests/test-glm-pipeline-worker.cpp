// Test-only CPU worker. Fixed synthetic model, per-run key file and loopback interfaces only.
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "llama-stage.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "stage-net.h"

#include <nlohmann/json.hpp>

#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

using namespace glm_rpc;
using json = nlohmann::json;

static deadline shortly() { return std::chrono::steady_clock::now() + std::chrono::seconds(30); }

static uint32_t integer(const std::string & value, uint32_t maximum) {
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || result > maximum) {
        throw std::invalid_argument("invalid pipeline fixture integer");
    }
    return result;
}

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void require_fixture(const std::string & path) {
    require(std::filesystem::file_size(path) <= (128ull << 20), "worker accepts only the small synthetic test model");
    gguf_context_ptr metadata(gguf_init_from_file(path.c_str(), {true, nullptr}));
    require(bool(metadata), "invalid fixture GGUF metadata");
    const int64_t name = gguf_find_key(metadata.get(), "general.name");
    require(name >= 0 && gguf_get_kv_type(metadata.get(), name) == GGUF_TYPE_STRING &&
            std::strcmp(gguf_get_val_str(metadata.get(), name), "LlamaGLM synthetic stage fixture v1") == 0,
            "worker refuses non-fixture models");
    for (const auto & field : std::map<std::string, uint32_t>{{"block_count", 6}, {"embedding_length", 128},
            {"context_length", 256}, {"expert_count", 16}, {"expert_used_count", 8}}) {
        const auto key = "glm5-next." + field.first;
        const int64_t id = gguf_find_key(metadata.get(), key.c_str());
        require(id >= 0 && gguf_get_kv_type(metadata.get(), id) == GGUF_TYPE_UINT32 &&
                gguf_get_val_u32(metadata.get(), id) == field.second, "unsupported synthetic fixture geometry");
    }
}

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "invalid pipeline worker arguments");
        }
        for (const auto & arg : args) {
            require(arg.first == "--rank" || arg.first == "--model" || arg.first == "--requests" ||
                    arg.first == "--output" || arg.first == "--next-port" || arg.first == "--key-file", "unknown pipeline worker argument");
        }
        const uint32_t rank = integer(args.at("--rank"), 2);
        const std::string model_path = args.at("--model");
        const std::string request_path = args.at("--requests");
        require_fixture(model_path);
        require(std::filesystem::file_size(request_path) <= 16384, "fixture requests exceed limit");
        std::ifstream requests_file(request_path);
        const auto requests = json::parse(requests_file);
        const std::vector<uint32_t> expected_chunks{1,3,4,17,31,64,11};
        require(requests.at("version") == 1 && requests.at("context") == 256 && requests.at("hidden") == 128 &&
                requests.at("vocab") == 128 && requests.at("chunks") == expected_chunks, "invalid fixture request schema");
        const auto & token_values = requests.at("tokens");
        require(token_values.is_array() && token_values.size() == 131, "invalid fixture token count");
        std::vector<int32_t> tokens;
        for (const auto & token : token_values) {
            require(token.is_number_integer(), "fixture tokens must be integers");
            if (token.is_number_unsigned()) { require(token.get<uint64_t>() < 128, "invalid fixture token ID"); }
            else { require(token.get<int64_t>() >= 0 && token.get<int64_t>() < 128, "invalid fixture token ID"); }
            tokens.push_back(token.get<int32_t>());
        }
        const uint32_t byte_order = 1;
        require(*reinterpret_cast<const uint8_t *>(&byte_order) == 1, "fixture worker requires little-endian IEEE binary32");
        const uint16_t next_port = args.count("--next-port") ? uint16_t(integer(args.at("--next-port"), 65535)) : 0;
        require(rank == 2 ? next_port == 0 : next_port != 0, "fixture next peer is missing");
        require(!std::filesystem::exists(args.at("--output")) && !std::filesystem::is_symlink(args.at("--output")),
                "fixture worker output must not already exist");
        struct key_file {
            secret value{};
            ~key_file() { erase_secret(value); }
        } credential;
        auto & psk = credential.value;
        require(std::filesystem::file_size(args.at("--key-file")) == psk.size(), "fixture key file must contain exactly 32 bytes");
        std::ifstream key_input(args.at("--key-file"), std::ios::binary);
        key_input.read(reinterpret_cast<char *>(psk.data()), psk.size());
        require(bool(key_input), "cannot read configured fixture key");
        key_input.close();

        common_init();
        common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
        llama_backend_init();
        auto params = llama_model_default_params();
        ggml_backend_dev_t cpu_only[] = {nullptr};
        params.devices = cpu_only;
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.lazy_mode = LLAMA_LAZY_MODE_OFF;
        params.progress_callback = [](float, void *) { return true; };
        llama_model_ptr model(llama_model_load_stage_from_file(model_path.c_str(), params, {1,rank,{2,4,6}}));
        require(bool(model), "owned fixture model failed to load");
        require(llama_model_n_layer(model.get()) == 6 && llama_model_n_embd(model.get()) == 128 &&
                llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 128, "fixture model dimensions changed");
        auto context_params = llama_stage_context_default_params();
        context_params.n_batch = 64;
        context_params.n_ubatch = 32;
        context_params.n_threads = context_params.n_threads_batch = 2;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        llama_context_ptr context(llama_init_stage_from_model(model.get(), context_params));
        require(bool(context), "owned fixture context failed to initialize");
        std::ofstream output(args.at("--output"), std::ios::binary);
        require(bool(output), "cannot create fixture worker output");
        digest identity; identity.fill(0x67);
        limits bounds;
        bounds.hidden = 128; bounds.rows = 64; bounds.context = 256;
        tcp_socket outgoing, incoming;
        std::unique_ptr<sender> tx;
        std::unique_ptr<receiver> rx;
        if (rank < 2) {
            outgoing = tcp_socket::connect({"127.0.0.1", next_port}, shortly());
            const auto peer = authenticate_client(outgoing, psk, {identity,rank,rank+1,channel::data}, shortly());
            tx = std::make_unique<sender>(derive_link_key(psk, peer), peer, bounds);
        }
        if (rank > 0) {
            auto listener = tcp_socket::listen({"127.0.0.1", 0});
            std::printf("GLM_PIPELINE_LISTEN %u %u\n", rank, unsigned(listener.local_endpoint().port));
            std::fflush(stdout);
            incoming = listener.accept(shortly());
            const auto peer = authenticate_server(incoming, psk, {identity,rank-1,rank,channel::data}, shortly());
            rx = std::make_unique<receiver>(derive_link_key(psk, peer), peer, bounds);
        }
        std::printf("GLM_PIPELINE_OWNED %u %u %u %llu\n", rank, 2*rank, 2*rank+2,
                static_cast<unsigned long long>(llama_model_size(model.get())));
        std::fflush(stdout);
        erase_secret(psk);
        size_t position = 0;
        for (size_t chunk = 0; chunk < expected_chunks.size(); ++chunk) {
            const uint32_t count = expected_chunks[chunk];
            common_batch batch(context.get());
            std::vector<float> input;
            if (rank == 0) {
                for (uint32_t row = 0; row < count; ++row) { batch.add(tokens.at(position + row), position + row, 0, false); }
            } else {
                const auto message = read_frame(incoming, *rx, shortly());
                session_id expected_session{}; expected_session[0] = 1;
                require(message.metadata.chunk == chunk && message.metadata.transaction == chunk + 1 &&
                        message.metadata.epoch == 1 && message.metadata.generation == 1 && message.metadata.session == expected_session &&
                        message.metadata.position == position && message.metadata.rows == count && message.metadata.fragments == 1 &&
                        message.payload.size() == count * 512 * sizeof(float), "fixture boundary coordinates differ");
                input.resize(count * 512);
                std::memcpy(input.data(), message.payload.data(), message.payload.size());
                for (uint32_t row = 0; row < count; ++row) {
                    const llama_pos pos = position + row;
                    batch.add_embd({input.data() + row * 512, 1, 512}, &pos, 0, rank == 2);
                }
            }
            require(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "owned stage decode failed");
            const float * values = rank == 2 ? llama_get_logits(context.get()) : llama_get_embeddings(context.get());
            require(values != nullptr, "owned stage did not return output");
            const size_t bytes = count * (rank == 2 ? 128 : 512) * sizeof(float);
            output.write(reinterpret_cast<const char *>(values), bytes);
            require(bool(output), "cannot write fixture worker output");
            if (rank < 2) {
                header metadata;
                metadata.sender = rank; metadata.receiver = rank + 1;
                metadata.epoch = metadata.generation = 1;
                metadata.transaction = chunk + 1; metadata.chunk = chunk;
                metadata.session[0] = 1; metadata.position = position;
                metadata.rows = count; metadata.hidden = 128; metadata.streams = 4;
                metadata.element = dtype::f32; metadata.total_bytes = bytes;
                write_frame(outgoing, *tx, metadata, {reinterpret_cast<const uint8_t *>(values), bytes}, shortly());
            }
            position += count;
        }
        output.close();
        require(bool(output), "cannot seal fixture worker output");
        std::printf("GLM_PIPELINE_DONE %u 131\n", rank);
        std::fflush(stdout);
        common_log_flush(common_log_main());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM pipeline fixture worker failed: %s\n", error.what());
        common_log_flush(common_log_main());
        return 1;
    }
}
