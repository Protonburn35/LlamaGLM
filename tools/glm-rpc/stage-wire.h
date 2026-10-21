#pragma once

#include "stage-crypto.h"

namespace glm_rpc {

constexpr size_t wire_header_bytes = 160;
constexpr size_t wire_mac_offset = 128;
constexpr uint32_t max_frame_bytes = 4u << 20;
constexpr uint32_t max_control_bytes = 16u << 10;
constexpr uint64_t max_chunk_bytes = 128ull << 20;

using wire_header = std::array<uint8_t, wire_header_bytes>;
using session_id = std::array<uint8_t, 16>;

enum class channel : uint8_t { data = 1, control = 2 };
enum class kind : uint16_t {
    hello = 1, capabilities, ready, begin, reset, boundary, state_ack,
    commit, abort, cancel, cancel_ack, heartbeat, error, commit_ack, commit_complete, terminal_result,
    sampled_result, sampling_setup, sampling_setup_ack,
};
enum class dtype : uint16_t { none = 0, f32 = 1 };

struct limits {
    uint32_t hidden = 4096;
    uint32_t streams = 4;
    uint32_t rows = 2048;
    uint32_t context = 1048576;
    uint32_t frame_bytes = max_frame_bytes;

    void validate() const;
};

struct binding {
    digest server_nonce{};
    digest client_nonce{};
    digest identity{}; // Canonical model/quant/split/precision/capability identity, agreed before use.
    uint32_t sender = 0;
    uint32_t receiver = 1;
    channel link = channel::data;
};

// Both nonces must be freshly generated for each connection; identity is never a secret.
secret derive_link_key(const secret & psk, const binding & peer);

struct header {
    kind type = kind::boundary;
    uint32_t sender = 0;
    uint32_t receiver = 1;
    uint64_t epoch = 0;
    uint64_t transaction = 0;
    uint64_t chunk = 0;
    uint64_t position = 0;
    session_id session{};
    uint64_t generation = 0;
    uint32_t rows = 0;
    uint32_t hidden = 0;
    uint16_t streams = 0;
    dtype element = dtype::none;
    uint32_t fragment = 0;
    uint32_t fragments = 1;
    uint32_t payload_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t sequence = 0;
};

// Connection-local directional state, owned by one writer/reader. Do not reuse across reconnects.
class sender {
public:
    sender(const secret & key, const binding & peer, const limits & bounds = {});
    ~sender();
    sender(const sender &) = delete;
    sender & operator=(const sender &) = delete;

    std::vector<uint8_t> encode(header metadata, bytes_view payload);

private:
    secret key;
    binding peer;
    limits bounds;
    uint64_t next_sequence = 1;
};

class receiver {
public:
    receiver(const secret & key, const binding & peer, const limits & bounds = {});
    ~receiver();
    receiver(const receiver &) = delete;
    receiver & operator=(const receiver &) = delete;

    // Inspect the fixed header BEFORE allocating/reading its bounded payload. This is not authentication.
    header inspect(const wire_header & raw);
    // Authenticate the complete frame BEFORE exposing its payload to a decoder/model or acknowledging it.
    // Any invalid header, sequence, length or MAC permanently invalidates this receiver.
    header authenticate(const wire_header & raw, bytes_view payload);

private:
    secret key;
    binding peer;
    limits bounds;
    uint64_t next_sequence = 1;
    bool failed = false;
};

} // namespace glm_rpc
