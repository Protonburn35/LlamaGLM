// Test-only CPU worker. Fixed synthetic model, per-run key file and loopback interfaces only.
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "llama-stage.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "stage-net.h"
#include "stage-identity.h"
#include "stage-boundary.h"
#include "stage-commit-barrier.h"
#include "stage-replies.h"
#include "glm-fixture-lock-quota.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>

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

static digest fingerprint_bytes(const void * data, size_t bytes) {
    return sha256({{static_cast<const uint8_t *>(data), bytes}});
}

static session_id parse_session(const std::string & value) {
    require(value.size() == 32, "fixture session requires 16 hexadecimal bytes");
    session_id result{};
    for (size_t i = 0; i < result.size(); ++i) {
        unsigned byte = 0;
        const auto parsed = std::from_chars(value.data()+2*i,value.data()+2*i+2,byte,16);
        require(parsed.ec == std::errc() && parsed.ptr == value.data()+2*i+2, "invalid fixture session byte");
        result[i] = uint8_t(byte);
    }
    require(result != session_id{}, "fixture session must be nonzero");
    return result;
}

struct epoch_plan {
    session_id session{};
    uint32_t generation = 1;
    uint16_t next_port = 0;
    std::array<uint16_t,3> controls{};
    bool drop_ack = false;
    uint32_t drop_chunk = 0;
};

static epoch_plan read_recovery_plan(uint32_t rank, const epoch_plan & previous) {
    // Test-controller pipe only. No production network configuration or secret is accepted here.
    std::array<char,1024> line{};
    require(bool(std::cin.getline(line.data(),line.size())), "missing/oversized fixture recovery plan");
    const auto value = json::parse(line.data());
    require(value.is_object() && value.size() == 6 && value.at("version") == 1, "invalid fixture recovery plan schema");
    const auto number = [&](const char * key, uint32_t maximum) {
        const auto & field = value.at(key);
        require(field.is_number_unsigned() && field.get<uint64_t>() <= maximum, "invalid fixture recovery plan integer");
        return field.get<uint32_t>();
    };
    epoch_plan result;
    result.session = parse_session(value.at("session").get<std::string>());
    result.generation = number("generation",UINT32_MAX);
    result.next_port = uint16_t(number("next_port",65535));
    result.controls[1] = uint16_t(number("control_b",65535)); result.controls[2] = uint16_t(number("control_c",65535));
    require(result.generation > previous.generation && result.session != previous.session &&
            (rank == 2 ? result.next_port == 0 : result.next_port != 0) &&
            (rank == 0 ? result.controls[1] && result.controls[2] : !result.controls[1] && !result.controls[2]),
            "fixture recovery requires fresh session, newer epoch/generation and complete topology");
    return result;
}

static tcp_io_stats sum_io(const tcp_io_stats & a, const tcp_io_stats & b) {
    const auto add = [](uint64_t x, uint64_t y) { return y > UINT64_MAX-x ? UINT64_MAX : x+y; };
    return {add(a.read_bytes,b.read_bytes),add(a.written_bytes,b.written_bytes),add(a.recv_calls,b.recv_calls),add(a.send_calls,b.send_calls),
        add(a.read_waits,b.read_waits),add(a.write_waits,b.write_waits),add(a.read_wait_ns,b.read_wait_ns),add(a.write_wait_ns,b.write_wait_ns),
        add(a.read_failures,b.read_failures),add(a.write_failures,b.write_failures),add(a.timeout_calls,b.timeout_calls)};
}

static json io_evidence(const tcp_io_stats & value) {
    return {{"read_bytes",value.read_bytes},{"written_bytes",value.written_bytes},{"recv_calls",value.recv_calls},{"send_calls",value.send_calls},
        {"read_waits",value.read_waits},{"write_waits",value.write_waits},{"read_wait_ns",value.read_wait_ns},{"write_wait_ns",value.write_wait_ns},
        {"read_failures",value.read_failures},{"write_failures",value.write_failures},{"timeout_calls",value.timeout_calls}};
}

struct fixture_control {
    tcp_socket socket;
    tcp_io_stats retired_io;
    std::unique_ptr<sender> tx;
    std::unique_ptr<receiver> rx;

    void bind(const secret & psk, const binding & forward, uint32_t local_rank, const limits & bounds) {
        auto outgoing = forward, incoming = forward;
        std::swap(incoming.sender, incoming.receiver);
        if (local_rank) { std::swap(outgoing, incoming); }
        struct scoped_secret {
            secret value;
            ~scoped_secret() { erase_secret(value); }
        } tx_key{derive_link_key(psk, outgoing)}, rx_key{derive_link_key(psk, incoming)};
        tx = std::make_unique<sender>(tx_key.value, outgoing, bounds);
        rx = std::make_unique<receiver>(rx_key.value, incoming, bounds);
    }
};

static void send_control(fixture_control & peer, const step_key & step, kind type,
        uint32_t sender_rank, uint32_t receiver_rank, bytes_view payload) {
    header metadata;
    metadata.type = type; metadata.sender = sender_rank; metadata.receiver = receiver_rank;
    metadata.session = step.stamp.session; metadata.epoch = step.stamp.epoch; metadata.generation = step.stamp.generation;
    metadata.transaction = step.transaction; metadata.chunk = step.chunk; metadata.position = step.position;
    metadata.payload_bytes = uint32_t(payload.size); metadata.total_bytes = payload.size;
    write_frame(peer.socket, *peer.tx, metadata, payload, shortly());
}

static void connect_controls(uint32_t rank, std::array<fixture_control, 3> & controls, tcp_socket & control_listener,
        const secret & psk, const digest & identity, const limits & bounds, const std::array<uint16_t, 3> & ports) {
    // Reconnect only after known global completion. Old directional codecs/counters are discarded.
    for (auto & link : controls) {
        const auto totals = sum_io(link.retired_io,link.socket.io_stats());
        link.socket.interrupt(); link = fixture_control{}; link.retired_io = totals;
    }
    if (rank == 0) {
        for (uint32_t remote : {1u, 2u}) {
            auto & link = controls[remote];
            const auto port = ports[remote];
            link.socket = tcp_socket::connect({"127.0.0.1", port}, shortly());
            const auto peer = authenticate_client(link.socket, psk, {identity,0,remote,channel::control}, shortly());
            link.bind(psk, peer, rank, bounds);
        }
    } else {
        auto & link = controls[0];
        link.socket = control_listener.accept(shortly());
        const auto peer = authenticate_server(link.socket, psk, {identity,0,rank,channel::control}, shortly());
        link.bind(psk, peer, rank, bounds);
    }
}

static step_key agree_begin(uint32_t rank, const step_key & proposed, std::array<fixture_control,3> & controls,
        const limits & bounds, bool committed_retry = false) {
    if (!rank) {
        const auto payload = encode_step_begin(proposed,bounds);
        for (uint32_t remote : {1u,2u}) { send_control(controls[remote],proposed,kind::begin,0,remote,{payload.data(),payload.size()}); }
        return proposed;
    }
    auto & link = controls[0];
    const auto frame = read_frame(link.socket,*link.rx,shortly());
    const begin_coordinates coordinates{proposed.stamp,proposed.transaction,proposed.chunk,proposed.position,proposed.rows};
    const auto agreed = decode_step_begin(frame.metadata,{frame.payload.data(),frame.payload.size()},coordinates,bounds);
    require(!committed_retry || agreed == proposed, "reconnected BEGIN changed the committed request key");
    return agreed;
}

static terminal_outcome greedy_outcome(const reply_lease & reply, const commit_certificate & certificate, const limits & bounds) {
    const auto bytes = reply->payload();
    const auto & step = reply->receipt().step;
    require(reply->receipt().rank == 2 && bytes.size == size_t(step.rows)*128*sizeof(float), "invalid terminal fixture logits");
    std::vector<llama_token_data> candidates(128);
    for (uint32_t token = 0; token < candidates.size(); ++token) {
        float score; std::memcpy(&score,bytes.data+((step.rows-1)*128+token)*sizeof(float),sizeof(float));
        candidates[token] = {llama_token(token),score,0};
    }
    llama_sampler_ptr sampler(llama_sampler_init_greedy());
    require(bool(sampler), "cannot create native terminal greedy sampler");
    llama_token_data_array native{candidates.data(),candidates.size(),-1,false};
    llama_sampler_apply(sampler.get(),&native);
    require(native.selected >= 0 && size_t(native.selected) < native.size, "native terminal sampler did not select a token");
    terminal_outcome result; result.step = step; result.certificate_digest = fingerprint_commit(certificate,bounds);
    result.vocabulary = 128; result.selected_token = uint32_t(native.data[native.selected].id);
    result.selected_logit = native.data[native.selected].logit; result.count = 8; result.row = step.rows-1;
    std::sort(candidates.begin(),candidates.end(),[](const auto & a, const auto & b) {
        return a.logit == b.logit ? a.id < b.id : a.logit > b.logit;
    });
    for (size_t i = 0; i < result.count; ++i) { result.top[i] = {uint32_t(candidates[i].id),candidates[i].logit}; }
    return result;
}

static json terminal_evidence(const terminal_outcome & result) {
    const auto bits = [](float value) { uint32_t encoded; std::memcpy(&encoded,&value,sizeof(encoded)); return encoded; };
    json top = json::array();
    for (size_t i = 0; i < result.count; ++i) { top.push_back({result.top[i].token,bits(result.top[i].logit)}); }
    return {{"token",result.selected_token},{"logit_bits",bits(result.selected_logit)},{"row",result.row},{"top",top}};
}

static commit_certificate finish_transaction(uint32_t rank, const step_key & step, const owner_receipt & receipt,
        owner_transactions & ledger, reply_cache & replies, std::array<fixture_control, 3> & controls,
        const limits & bounds, bool drop_ack, std::optional<terminal_outcome> & terminal, size_t & terminal_transfers) {
    commit_certificate decision;
    if (rank == 0) {
        commit_barrier barrier(step, bounds);
        barrier.record_applied(0, receipt);
        for (uint32_t remote : {1u, 2u}) {
            auto & link = controls[remote];
            const auto frame = read_frame(link.socket, *link.rx, shortly());
            barrier.record_applied(remote, decode_applied_ack(frame.metadata,
                {frame.payload.data(), frame.payload.size()}, step, bounds));
        }
        decision = barrier.decision();
        const auto encoded = encode_commit_certificate(decision, bounds);
        for (uint32_t remote : {1u, 2u}) {
            send_control(controls[remote], step, kind::commit, 0, remote, {encoded.data(), encoded.size()});
        }
        const auto local = ledger.commit(decision);
        require(replies.commit(decision)->receipt() == local, "A committed reply differs from actual owner state");
        barrier.record_committed(0, make_commit_ack(local, decision, bounds));
        const auto fingerprint = fingerprint_commit(decision, bounds);
        for (uint32_t remote : {1u, 2u}) {
            auto & link = controls[remote];
            const auto frame = read_frame(link.socket, *link.rx, shortly());
            barrier.record_committed(remote, decode_committed_ack(frame.metadata,
                {frame.payload.data(), frame.payload.size()}, step, fingerprint, bounds));
        }
        const auto completion = encode_commit_completion(barrier.committed(), bounds);
        for (uint32_t remote : {1u, 2u}) {
            send_control(controls[remote], step, kind::commit_complete, 0, remote, {completion.data(), completion.size()});
        }
        auto & link = controls[2];
        const auto result = read_frame(link.socket,*link.rx,shortly());
        terminal = decode_terminal_outcome(result.metadata,{result.payload.data(),result.payload.size()},step,fingerprint,128,8,bounds);
        ++terminal_transfers;
    } else {
        auto & link = controls[0];
        const auto encoded = encode_applied_receipt(receipt, bounds);
        send_control(link, step, kind::state_ack, rank, 0, {encoded.data(), encoded.size()});
        const auto command = read_frame(link.socket, *link.rx, shortly());
        decision = decode_commit_decision(command.metadata, {command.payload.data(), command.payload.size()}, step, bounds);
        const auto local = ledger.commit(decision);
        require(replies.commit(decision)->receipt() == local, "committed reply differs from actual owner state");
        const auto acknowledged = make_commit_ack(local, decision, bounds);
        if (drop_ack) { throw std::runtime_error("injected lost B commit acknowledgement after local commit"); }
        const auto ack = encode_commit_ack(acknowledged, bounds);
        send_control(link, step, kind::commit_ack, rank, 0, {ack.data(), ack.size()});
        const auto complete = read_frame(link.socket, *link.rx, shortly());
        decode_commit_completion(complete.metadata, {complete.payload.data(), complete.payload.size()},
            step, acknowledged.certificate_digest, bounds);
        if (rank == 2) {
            if (!terminal || !(terminal->step == step) || terminal->certificate_digest != acknowledged.certificate_digest) {
                terminal = greedy_outcome(replies.committed_reply(step),decision,bounds);
            }
            const auto payload = encode_terminal_outcome(*terminal,bounds);
            send_control(link,step,kind::terminal_result,2,0,{payload.data(),payload.size()});
            ++terminal_transfers;
        }
    }
    return decision;
}

static execution_identity fixture_identity(const std::string & path, bool locked) {
    require(std::filesystem::file_size(path) <= (128ull << 20), "worker accepts only the small synthetic test model");
    gguf_context_ptr metadata(gguf_init_from_file(path.c_str(), {true, nullptr}));
    require(bool(metadata), "invalid fixture GGUF metadata");
    const int64_t name = gguf_find_key(metadata.get(), "general.name");
    require(name >= 0 && gguf_get_kv_type(metadata.get(), name) == GGUF_TYPE_STRING &&
            std::strcmp(gguf_get_val_str(metadata.get(), name), locked ? "LlamaGLM synthetic locked host fixture v1" :
                "LlamaGLM synthetic stage fixture v1") == 0,
            "worker refuses non-fixture models");
    for (const auto & field : std::map<std::string, uint32_t>{{"block_count", 6}, {"embedding_length", 128},
            {"context_length", 256}, {"expert_count", 16}, {"expert_used_count", 8}}) {
        const auto key = "glm5-next." + field.first;
        const int64_t id = gguf_find_key(metadata.get(), key.c_str());
        require(id >= 0 && gguf_get_kv_type(metadata.get(), id) == GGUF_TYPE_UINT32 &&
                gguf_get_val_u32(metadata.get(), id) == field.second, "unsupported synthetic fixture geometry");
    }
    llama_stage_schema_report report{1, 0, 0, 0, 0};
    require(llama_model_validate_stage_schema(metadata.get(), &report), "complete fixture tensor schema refused");
    const uint64_t header_bytes = gguf_get_data_offset(metadata.get());
    require(header_bytes >= 24 && header_bytes <= (32ull << 20), "fixture header exceeds identity limit");
    std::vector<uint8_t> header_data(size_t(header_bytes), 0);
    std::ifstream header_file(path, std::ios::binary);
    header_file.read(reinterpret_cast<char *>(header_data.data()), header_data.size());
    require(bool(header_file), "cannot read complete fixture header identity");
    execution_identity value;
    value.source_revision = compiled_source_revision();
    value.model_repository = "LlamaGLM/synthetic";
    value.model_revision = value.source_revision;
    value.quant = "synthetic-f32";
    value.shards.push_back({0, "model.gguf", std::filesystem::file_size(path), header_bytes,
        sha256({{header_data.data(), header_data.size()}})});
    value.tensors = report.tensors; value.payload_bytes = report.payload_bytes;
    value.target_layers = report.target_layers; value.nextn_layers = report.nextn_layers;
    value.experts = 16; value.layer_ends = locked ? std::array<uint32_t, 3>{2,3,6} : std::array<uint32_t, 3>{2,4,6};
    value.bounds.hidden = 128; value.bounds.rows = 64; value.bounds.context = 256;
    value.bounds.frame_bytes = 64u << 10;
    return value;
}

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "invalid pipeline worker arguments");
        }
        for (const auto & arg : args) {
            require(arg.first == "--rank" || arg.first == "--model" || arg.first == "--requests" ||
                    arg.first == "--output" || arg.first == "--next-port" || arg.first == "--key-file" ||
                    arg.first == "--storage" || arg.first == "--session" || arg.first == "--generation" ||
                    arg.first == "--control-b" || arg.first == "--control-c" || arg.first == "--drop-commit-ack" ||
                    arg.first == "--recover-once" || arg.first == "--drop-commit-at",
                    "unknown pipeline worker argument");
        }
        const auto storage_mode = args.count("--storage") ? args.at("--storage") : "buffered";
        require(storage_mode == "buffered" || storage_mode == "streamed" || storage_mode == "locked", "unknown fixture storage profile");
        const bool locked = storage_mode == "locked";
        const bool streamed = storage_mode != "buffered";
        const uint32_t rank = integer(args.at("--rank"), 2);
        const bool transactional = args.count("--session") != 0;
        const bool recover_once = args.count("--recover-once") && integer(args.at("--recover-once"),1);
        require(!recover_once || transactional, "retained recovery requires transactional fixture mode");
        const bool drop_ack = args.count("--drop-commit-ack") && integer(args.at("--drop-commit-ack"), 1);
        const auto drop_chunk = args.count("--drop-commit-at") ? integer(args.at("--drop-commit-at"),6) : 0;
        require(!args.count("--drop-commit-at") || drop_ack, "failure position requires lost-ACK injection");
        require(!drop_ack || (transactional && rank == 1), "lost-ACK injection is limited to transactional fixture B");
        require(!transactional || locked, "transaction fixture requires the small locked-host profile");
        session_id session{};
        uint32_t generation = 1;
        if (transactional) {
            session = parse_session(args.at("--session"));
            generation = integer(args.at("--generation"), UINT32_MAX);
            require(generation > 0 && session != session_id{}, "fixture session/generation must be nonzero");
        }
        const std::string model_path = args.at("--model");
        const std::string request_path = args.at("--requests");
        common_init();
        common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
        llama_backend_init();
        const auto agreed = fixture_identity(model_path, locked);
        const auto identity = identity_digest(agreed);
        require(std::filesystem::file_size(request_path) <= 16384, "fixture requests exceed limit");
        std::ifstream requests_file(request_path);
        const auto requests = json::parse(requests_file);
        const std::vector<uint32_t> expected_chunks{1,3,4,17,31,64,11};
        require(requests.at("version") == 1 && requests.at("context") == 256 && requests.at("hidden") == 128 &&
                requests.at("vocab") == 128 && requests.at("chunks") == expected_chunks &&
                requests.at("layer_ends") == json(agreed.layer_ends) &&
                requests.at("microbatch") == (streamed ? 1 : 32), "invalid fixture request schema/profile");
        std::vector<int32_t> tokens;
        if (!transactional || rank == 0) {
            const auto & token_values = requests.at("tokens");
            require(token_values.is_array() && token_values.size() == 131, "invalid fixture token count");
            for (const auto & token : token_values) {
                require(token.is_number_integer(), "fixture tokens must be integers");
                if (token.is_number_unsigned()) { require(token.get<uint64_t>() < 128, "invalid fixture token ID"); }
                else { require(token.get<int64_t>() >= 0 && token.get<int64_t>() < 128, "invalid fixture token ID"); }
                tokens.push_back(token.get<int32_t>());
            }
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

        auto params = llama_model_default_params();
        ggml_backend_dev_t cpu_only[] = {nullptr};
        params.devices = cpu_only;
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.lazy_mode = LLAMA_LAZY_MODE_OFF;
        params.progress_callback = [](float, void *) { return true; };
        const bool desktop_source = streamed && rank != 1;
        const bool preloaded = locked && rank == 1;
        params.use_extra_bufts = !locked;
        std::optional<glm_fixture_lock_quota> quota;
        if (preloaded) { quota.emplace(); }
        const llama_stage_storage_params storage{1,8,16u<<20,16u<<20,64,2};
        const llama_stage_preload_params preload{2,8,2u<<20,8u<<20,4u<<20,64,0};
        const auto & ends = agreed.layer_ends;
        const llama_stage_model_params stage{1,rank,{ends[0],ends[1],ends[2]}};
        size_t model_loads = 0;
        llama_model_ptr model(desktop_source ?
            llama_model_load_stage_streamed_from_file(model_path.c_str(), params, stage, storage) : preloaded ?
            llama_model_load_stage_preloaded_from_file(model_path.c_str(), params, stage, preload) :
            llama_model_load_stage_from_file(model_path.c_str(), params, stage));
        require(bool(model), "owned fixture model failed to load");
        ++model_loads;
        require(llama_model_n_layer(model.get()) == 6 && llama_model_n_embd(model.get()) == 128 &&
                llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 128, "fixture model dimensions changed");
        const auto report_storage = [&](const char * phase) {
            llama_stage_storage_report report{}; report.version = 1;
            const bool has_source = llama_model_stage_storage_report(model.get(), &report);
            require(has_source == (desktop_source || preloaded), "fixture worker has the wrong storage provider");
            json evidence{{"version",1}, {"rank",rank}, {"phase",phase}, {"microbatch",streamed ? 1 : 32},
                {"mode",desktop_source ? "desktop-streamed" : preloaded ? "host-preloaded" : "buffered"}};
            if (has_source) {
                require(report.streamed_tensors > 0 && report.routed_logical_bytes > 0 &&
                        report.model_buffer_bytes < llama_model_size(model.get()) && report.host_reserved_bytes <= storage.host_cache_bytes,
                        "streamed worker allocated full experts or exceeded its host cap");
                const bool startup = std::strcmp(phase, "startup") == 0;
                if (desktop_source) {
                    require(startup ? (report.native_read_requests == 0 && report.native_read_bytes == 0 && report.host_reserved_bytes == 0) :
                            (report.native_read_requests > 0 && report.native_read_bytes > 0), "unexpected streamed worker source I/O");
                }
                evidence["streamed_tensors"] = report.streamed_tensors;
                evidence["routed_logical_bytes"] = report.routed_logical_bytes;
                evidence["model_buffer_bytes"] = report.model_buffer_bytes;
                evidence["host_reserved_bytes"] = report.host_reserved_bytes;
                evidence["native_read_requests"] = report.native_read_requests;
                evidence["native_read_bytes"] = report.native_read_bytes;
                if (preloaded) {
                    llama_stage_preload_report experts{}; experts.version = 1;
                    llama_stage_host_report host{}; host.version = 1;
                    require(llama_model_stage_preload_report(model.get(), &experts) && llama_model_stage_host_report(model.get(), &host) &&
                            experts.expert_slabs == 48 && !experts.open_source_files && !experts.full_residency_validated &&
                            experts.payload_bytes == report.routed_logical_bytes && experts.startup_read_requests == 48 &&
                            experts.startup_read_bytes == report.routed_logical_bytes && report.native_read_requests == 48 &&
                            report.native_read_bytes == report.routed_logical_bytes && host.native_payload_locks_verified &&
                            !host.full_residency_validated && !host.pending_buffers && !host.failures && host.tensor_buffers &&
                            host.reserved_page_bytes > experts.locked_bytes && host.reserved_page_bytes <= preload.host_page_bytes &&
                            (startup || !std::filesystem::exists(model_path)), "locked middle worker lost its preload/host policy or retained a source");
                    evidence["expert_slabs"] = experts.expert_slabs;
                    evidence["locked_expert_bytes"] = experts.locked_bytes;
                    evidence["host_page_bytes"] = host.reserved_page_bytes;
                    evidence["peak_host_page_bytes"] = host.peak_reserved_page_bytes;
                    evidence["host_buffers"] = host.tensor_buffers;
                    evidence["native_locks_verified"] = host.native_payload_locks_verified;
                    evidence["full_residency_validated"] = host.full_residency_validated;
                    evidence["open_source_files"] = experts.open_source_files;
                    evidence["source_removed"] = !std::filesystem::exists(model_path);
                }
            }
            std::printf("GLM_PIPELINE_STORAGE %s\n", evidence.dump().c_str());
            std::fflush(stdout);
        };
        report_storage("startup");
        auto context_params = llama_stage_context_default_params_for_model(model.get());
        context_params.n_batch = 64;
        context_params.n_ubatch = streamed ? 1 : 32;
        context_params.n_threads = context_params.n_threads_batch = 2;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        epoch_plan initial{session,generation,next_port,{},drop_ack,drop_chunk};
        if (transactional && rank == 0) {
            initial.controls[1] = uint16_t(integer(args.at("--control-b"),65535));
            initial.controls[2] = uint16_t(integer(args.at("--control-c"),65535));
        }
        llama_stage_host_report baseline{}; baseline.version = 1;
        if (preloaded) { require(llama_model_stage_host_report(model.get(),&baseline), "missing baseline host report"); }
        size_t contexts_created = 0;
        bool epoch_invalidated = false;
        // Fixed fixture response journal survives context reset. It holds only fingerprints
        // of already globally completed output; it is not a prefix cache or durable snapshot.
        std::array<digest,7> published_fingerprints{};
        size_t published_chunks = 0, published_rows = 0;
        const auto execute_epoch = [&](const epoch_plan & plan) {
            const auto epoch_started = std::chrono::steady_clock::now();
            const auto & session = plan.session;
            const auto generation = plan.generation;
            const auto next_port = plan.next_port;
            const auto & control_ports = plan.controls;
            const auto drop_ack = plan.drop_ack;
            llama_context_ptr context(llama_init_stage_from_model(model.get(), context_params));
            require(bool(context), "owned fixture context failed to initialize");
            ++contexts_created;
            std::ofstream output(args.at("--output"), std::ios::binary | (published_rows ? std::ios::app : std::ios::trunc));
            require(bool(output), "cannot create fixture worker output");
            const auto bounds = agreed.bounds;
            tcp_socket outgoing, incoming;
            std::unique_ptr<sender> tx;
            std::unique_ptr<receiver> rx;
            std::array<fixture_control, 3> controls;
            tcp_socket control_listener;
            if (rank < 2) {
                outgoing = tcp_socket::connect({"127.0.0.1", next_port}, shortly());
                const auto peer = authenticate_client(outgoing, psk, {identity,rank,rank+1,channel::data}, shortly());
                tx = std::make_unique<sender>(derive_link_key(psk, peer), peer, bounds);
            }
            if (rank > 0) {
                auto listener = tcp_socket::listen({"127.0.0.1", 0});
                if (transactional) {
                    control_listener = tcp_socket::listen({"127.0.0.1", 0});
                    std::printf("GLM_PIPELINE_CONTROL %u %u\n", rank, unsigned(control_listener.local_endpoint().port));
                }
                std::printf("GLM_PIPELINE_LISTEN %u %u\n", rank, unsigned(listener.local_endpoint().port));
                std::fflush(stdout);
                incoming = listener.accept(shortly());
                const auto peer = authenticate_server(incoming, psk, {identity,rank-1,rank,channel::data}, shortly());
                rx = std::make_unique<receiver>(derive_link_key(psk, peer), peer, bounds);
            }
            if (transactional) { connect_controls(rank, controls, control_listener, psk, identity, bounds, control_ports); }
            std::printf("GLM_PIPELINE_OWNED %u %u %u %llu\n", rank, rank ? ends[rank-1] : 0, ends[rank],
                    static_cast<unsigned long long>(llama_model_size(model.get())));
            std::fflush(stdout);
            if (!transactional) { erase_secret(psk); }
            std::optional<owner_transactions> ledger;
            std::optional<reply_cache> replies;
            const epoch_stamp stamp{identity,session,generation,generation};
            if (transactional) {
                ledger.emplace(rank, identity, bounds); ledger->fresh_context(stamp);
                replies.emplace(rank, stamp, bounds, reply_limits{256u<<10,preloaded ? 256u<<10 : 0,3,
                    preloaded ? reply_memory::locked : reply_memory::pageable});
            }
            json commits = json::array();
            json terminal_results = json::array();
            std::optional<terminal_outcome> terminal;
            size_t terminal_transfers = 0;
            size_t executions = 0, committed_retries = 0, network_retries = 0;
            size_t replayed_rows = 0, new_output_rows = 0;
            uint64_t compute_ns = 0;
            size_t position = 0;
            try {
            for (size_t chunk = 0; chunk < expected_chunks.size(); ++chunk) {
                const uint32_t count = expected_chunks[chunk];
                header metadata;
                metadata.epoch = metadata.generation = 1;
                metadata.transaction = chunk + 1; metadata.chunk = chunk;
                metadata.session[0] = 1; metadata.position = position;
                metadata.rows = count; metadata.hidden = 128; metadata.streams = 4;
                metadata.element = dtype::f32;
                step_key step{stamp,chunk+1,chunk+1,position,count,
                    transactional && rank ? digest{} : fingerprint_bytes(tokens.data()+position,count*sizeof(int32_t))};
                if (transactional) {
                    step = agree_begin(rank,step,controls,bounds);
                    metadata.session = session; metadata.epoch = metadata.generation = generation; metadata.chunk = step.chunk;
                }
                common_batch batch(context.get());
                std::vector<float> input;
                if (rank == 0) {
                    for (uint32_t row = 0; row < count; ++row) { batch.add(tokens.at(position + row), position + row, 0, false); }
                } else {
                    metadata.sender = rank - 1; metadata.receiver = rank;
                    input = receive_boundary(incoming, *rx, metadata, bounds, uint64_t(bounds.rows) * 512 * sizeof(float), shortly());
                    for (uint32_t row = 0; row < count; ++row) {
                        const llama_pos pos = position + row;
                        batch.add_embd({input.data() + row * 512, 1, 512}, &pos, 0, rank == 2);
                    }
                }
                const auto input_digest = rank ? fingerprint_bytes(input.data(), input.size() * sizeof(float)) : step.request_digest;
                if (ledger) { require(ledger->prepare(step, input_digest) == step_disposition::execute, "fixture attempted duplicate graph execution"); }
                const auto compute_started = std::chrono::steady_clock::now();
                const int status = llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
                compute_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-compute_started).count());
                require(status == 0, "owned stage decode failed");
                ++executions;
                const float * values = rank == 2 ? llama_get_logits(context.get()) : llama_get_embeddings(context.get());
                require(values != nullptr, "owned stage did not return output");
                const size_t bytes = count * (rank == 2 ? 128 : 512) * sizeof(float);
                std::optional<owner_receipt> receipt;
                reply_lease retained;
                if (ledger) {
                    receipt = ledger->applied(step, fingerprint_bytes(values, bytes));
                    retained = replies->retain_applied(*receipt, {reinterpret_cast<const uint8_t *>(values),bytes});
                    if (preloaded) { retained->verify_locks(); }
                }
                if (rank < 2) {
                    metadata.sender = rank; metadata.receiver = rank + 1;
                    send_boundary(outgoing, *tx, metadata, bounds, values, count * 512, shortly());
                }
                if (transactional) {
                    const auto decision = finish_transaction(rank, step, *receipt, *ledger, *replies, controls, bounds,
                        drop_ack && chunk == plan.drop_chunk,terminal,terminal_transfers);
                    if (rank != 1) { require(bool(terminal), "missing committed terminal result"); terminal_results.push_back(terminal_evidence(*terminal)); }
                    const auto committed_hash = fingerprint_commit(decision, bounds);
                    std::string encoded_hash;
                    for (const auto byte : committed_hash) {
                        encoded_hash += "0123456789abcdef"[byte >> 4]; encoded_hash += "0123456789abcdef"[byte & 15];
                    }
                    commits.push_back(encoded_hash);
                    require(ledger->prepare(step, input_digest) == step_disposition::committed &&
                            ledger->committed_position() == position + count, "committed retry lost state or requested execution");
                    require(replies->committed_reply(step) == retained &&
                            fingerprint_bytes(retained->payload().data, retained->payload().size) == receipt->output_digest,
                            "committed retry lost immutable output bytes");
                    ++committed_retries;
                    if (chunk == 0) {
                        // All owners have a known complete first commit. Fresh control handshakes
                        // reset connection counters; semantic dedup and cached bytes must survive.
                        connect_controls(rank, controls, control_listener, psk, identity, bounds, control_ports);
                        agree_begin(rank,step,controls,bounds,true);
                        auto duplicate = metadata;
                        auto repeated_input = step.request_digest;
                        if (rank) {
                            duplicate.sender = rank-1; duplicate.receiver = rank;
                            input = receive_boundary(incoming, *rx, duplicate, bounds, uint64_t(bounds.rows)*512*sizeof(float), shortly());
                            repeated_input = fingerprint_bytes(input.data(), input.size()*sizeof(float));
                        }
                        require(ledger->prepare(step, repeated_input) == step_disposition::committed,
                                "reconnected committed request attempted graph execution");
                        const auto replay = replies->committed_reply(step);
                        require(replay == retained && fingerprint_bytes(replay->payload().data,replay->payload().size) == receipt->output_digest,
                                "reconnected retry lost the exact immutable reply");
                        if (rank < 2) {
                            duplicate.sender = rank; duplicate.receiver = rank+1;
                            send_boundary_bytes(outgoing, *tx, duplicate, bounds, replay->payload(), shortly());
                        }
                        const auto original_terminal = terminal;
                        const auto repeated = finish_transaction(rank, step, replay->receipt(), *ledger, *replies, controls, bounds,
                            false,terminal,terminal_transfers);
                        if (rank != 1) { require(encode_terminal_outcome(*terminal,bounds) == encode_terminal_outcome(*original_terminal,bounds),
                            "committed terminal retry changed its sampled outcome"); }
                        require(fingerprint_commit(repeated,bounds) == committed_hash && ledger->committed_position() == position+count,
                                "reconnected retry changed its complete certificate or committed position");
                        ++network_retries;
                    }
                }
                const auto output_bytes = retained ? retained->payload() : bytes_view{reinterpret_cast<const uint8_t *>(values),bytes};
                if (transactional && chunk < published_chunks) {
                    require(receipt->output_digest == published_fingerprints[chunk] && position+count <= published_rows &&
                            replayed_rows == position, "replayed committed response history changed");
                    replayed_rows += count;
                } else {
                    if (transactional) {
                        require(chunk == published_chunks && position == published_rows, "response cursor skipped committed history");
                    }
                    output.write(reinterpret_cast<const char *>(output_bytes.data), output_bytes.size);
                    require(bool(output), "cannot write fixture worker output");
                    if (transactional) {
                        published_fingerprints[published_chunks++] = receipt->output_digest;
                        published_rows += count; new_output_rows += count;
                    }
                }
                position += count;
            }
            } catch (...) {
                if (ledger) {
                    replies->invalidate();
                    ledger->invalidate();
                    require(llama_context_stage_invalidate(context.get()) && !llama_get_logits(context.get()) &&
                            !llama_get_embeddings(context.get()), "uncertain fixture context retained output");
                    outgoing.interrupt(); incoming.interrupt();
                    for (auto & link : controls) { link.socket.interrupt(); }
                    std::printf("GLM_PIPELINE_INVALIDATED %u\n", rank); std::fflush(stdout);
                    epoch_invalidated = true;
                }
                throw;
            }
            erase_secret(psk);
            output.close();
            require(bool(output), "cannot seal fixture worker output");
            report_storage("complete");
            if (ledger) {
                const auto retained = replies->stats();
                require(!retained.failures && !retained.refusals && retained.entries == 1 &&
                        retained.payload_bytes > 0 && retained.peak_payload_bytes <= (256u<<10) &&
                        retained.peak_page_bytes <= (256u<<10), "worker reply capacity or lifetime accounting failed");
                const json evidence{{"rank",rank}, {"generation",generation}, {"executions",executions}, {"model_loads",model_loads}, {"contexts_created",contexts_created},
                    {"committed_retries",committed_retries}, {"committed_position",ledger->committed_position()},
                    {"network_retries",network_retries},
                    {"terminal_results",terminal_results}, {"terminal_payload_bytes",terminal_transfers*terminal_wire_bytes},
                    {"replayed_rows",replayed_rows}, {"new_output_rows",new_output_rows}, {"published_rows",published_rows},
                    {"certificates",commits}, {"reply_bytes",retained.payload_bytes}, {"peak_reply_bytes",retained.peak_payload_bytes},
                    {"reply_page_bytes",retained.page_bytes}, {"peak_reply_page_bytes",retained.peak_page_bytes},
                    {"full_residency_validated",false}};
                std::printf("GLM_PIPELINE_TRANSACTIONS %s\n", evidence.dump().c_str());
            }
            llama_stage_bank_report bank{}; bank.version = 1;
            require(llama_context_stage_bank_report(context.get(), &bank) == (desktop_source || preloaded), "wrong fixture bank report source");
            if (desktop_source || preloaded) {
                require(bank.device_banks == 0 && bank.cpu_banks == (rank == 0 ? ends[0] - 1 : ends[rank] - ends[rank-1]) &&
                        bank.buffer_bytes <= (preloaded ? preload.context_bank_bytes : storage.context_bank_bytes) &&
                        bank.materializations == uint64_t(bank.cpu_banks) * 131 && bank.slot_hits > 0 && bank.failures == 0 &&
                        bank.slot_hits + bank.slot_misses == bank.materializations * 8 && bank.copied_slabs == bank.slot_misses * 3,
                        "streamed worker did not reuse complete backend expert groups or reported inconsistent counters");
                const json evidence{{"rank",rank}, {"cpu_banks",bank.cpu_banks}, {"device_banks",bank.device_banks},
                    {"buffer_bytes",bank.buffer_bytes}, {"materializations",bank.materializations}, {"hits",bank.slot_hits},
                    {"misses",bank.slot_misses}, {"copied_slabs",bank.copied_slabs}, {"copied_bytes",bank.copied_bytes}, {"failures",bank.failures}};
                std::printf("GLM_PIPELINE_BANK %s\n", evidence.dump().c_str());
            }
            json control_io = json::array();
            for (const auto & link : controls) { control_io.push_back(io_evidence(sum_io(link.retired_io,link.socket.io_stats()))); }
            const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-epoch_started).count();
            const json performance{{"rank",rank},{"generation",plan.generation},{"compute_ns",compute_ns},{"epoch_wall_ns",wall_ns},
                {"data_in",io_evidence(incoming.io_stats())},{"data_out",io_evidence(outgoing.io_stats())},{"control",control_io}};
            std::printf("GLM_PIPELINE_IO %s\n",performance.dump().c_str());
            std::printf("GLM_PIPELINE_DONE %u 131\n", rank);
            std::fflush(stdout);
        };
        try { execute_epoch(initial); }
        catch (const std::exception &) {
            if (!recover_once || contexts_created != 1 || !epoch_invalidated) { throw; }
            // execute_epoch has unwound: failed context, queues and reply owners are gone.
            if (preloaded) {
                llama_stage_host_report released{}; released.version = 1;
                require(llama_model_stage_host_report(model.get(),&released) && released.native_payload_locks_verified &&
                        !released.pending_buffers && released.tensor_buffers == baseline.tensor_buffers &&
                        released.reserved_page_bytes == baseline.reserved_page_bytes && !std::filesystem::exists(model_path),
                        "failed B context was not fully released against retained source-deleted weights");
            }
            std::printf("GLM_PIPELINE_RESET_READY %u\n",rank); std::fflush(stdout);
            const auto next = read_recovery_plan(rank,initial);
            std::printf("GLM_PIPELINE_RETAINED_RESET %u %u %u\n",rank,initial.generation,next.generation); std::fflush(stdout);
            execute_epoch(next);
        }
        common_log_flush(common_log_main());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM pipeline fixture worker failed: %s\n", error.what());
        common_log_flush(common_log_main());
        return 1;
    }
}
