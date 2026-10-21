#include "stage-wire.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace glm_rpc {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "GLM transport requires IEEE 754 binary32");
namespace {

constexpr uint8_t magic[8] = {'L', 'L', 'G', 'L', 'M', 'R', 'P', 'C'};
constexpr uint8_t key_domain[] = "LlamaGLM RPC directional key v1";
constexpr uint8_t frame_domain[] = "LlamaGLM RPC frame v1";

void put(uint8_t * output, uint64_t value, size_t size) {
    for (size_t i = 0; i < size; ++i) { output[i] = uint8_t(value >> (8 * i)); }
}

uint64_t get(const wire_header & input, size_t offset, size_t size) {
    uint64_t value = 0;
    for (size_t i = 0; i < size; ++i) { value |= uint64_t(input[offset + i]) << (8 * i); }
    return value;
}

template <size_t N>
bool zero(const std::array<uint8_t, N> & value) {
    return std::all_of(value.begin(), value.end(), [](uint8_t byte) { return byte == 0; });
}

void validate_binding(const binding & peer) {
    if (peer.sender > 2 || peer.receiver > 2 || peer.sender == peer.receiver ||
            (peer.link != channel::data && peer.link != channel::control) ||
            zero(peer.server_nonce) || zero(peer.client_nonce) || zero(peer.identity)) {
        throw std::invalid_argument("invalid GLM peer binding");
    }
}

void validate(const header & value, const binding & peer, const limits & bounds) {
    if (value.sender != peer.sender || value.receiver != peer.receiver || value.sequence == 0 ||
            value.sequence == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("invalid GLM peer or exhausted sequence");
    }
    switch (value.type) {
        case kind::boundary:
            break;
        case kind::hello: case kind::capabilities: case kind::ready: case kind::begin:
        case kind::reset: case kind::state_ack: case kind::commit: case kind::abort:
        case kind::cancel: case kind::cancel_ack: case kind::heartbeat: case kind::error:
        case kind::commit_ack: case kind::commit_complete: case kind::terminal_result:
            if (peer.link != channel::control || value.element != dtype::none || value.rows || value.hidden || value.streams ||
                    value.fragment != 0 || value.fragments != 1 || value.payload_bytes > max_control_bytes ||
                    value.total_bytes != value.payload_bytes) {
                throw std::invalid_argument("invalid or oversized GLM control frame");
            }
            return;
        default:
            throw std::invalid_argument("unknown GLM frame kind");
    }
    if (peer.link != channel::data || value.element != dtype::f32 || value.hidden != bounds.hidden ||
            value.streams != bounds.streams || value.rows == 0 || value.rows > bounds.rows ||
            value.position >= bounds.context || value.rows > bounds.context - value.position ||
            value.epoch == 0 || value.transaction == 0 || value.generation == 0 || zero(value.session)) {
        throw std::invalid_argument("invalid GLM boundary shape, precision or session coordinates");
    }
    const uint64_t expected = uint64_t(value.rows) * value.hidden * value.streams * sizeof(float);
    if (value.total_bytes != expected || expected > max_chunk_bytes) {
        throw std::invalid_argument("invalid GLM boundary byte count");
    }
    const uint64_t count = (expected + bounds.frame_bytes - 1) / bounds.frame_bytes;
    if (value.fragments != count || value.fragment >= count) {
        throw std::invalid_argument("invalid GLM boundary fragment coordinates");
    }
    const uint64_t offset = uint64_t(value.fragment) * bounds.frame_bytes;
    if (value.payload_bytes != std::min<uint64_t>(bounds.frame_bytes, expected - offset)) {
        throw std::invalid_argument("invalid GLM boundary fragment length");
    }
}

wire_header serialize(const header & value) {
    wire_header raw{};
    std::copy(std::begin(magic), std::end(magic), raw.begin());
    put(raw.data() + 8, 1, 2);
    put(raw.data() + 10, 0, 2);
    put(raw.data() + 12, wire_header_bytes, 2);
    put(raw.data() + 14, uint16_t(value.type), 2);
    put(raw.data() + 16, value.sender, 4);
    put(raw.data() + 20, value.receiver, 4);
    put(raw.data() + 24, value.epoch, 8);
    put(raw.data() + 32, value.transaction, 8);
    put(raw.data() + 40, value.chunk, 8);
    put(raw.data() + 48, value.position, 8);
    std::copy(value.session.begin(), value.session.end(), raw.begin() + 56);
    put(raw.data() + 72, value.generation, 8);
    put(raw.data() + 80, value.rows, 4);
    put(raw.data() + 84, value.hidden, 4);
    put(raw.data() + 88, value.streams, 2);
    put(raw.data() + 90, uint16_t(value.element), 2);
    put(raw.data() + 92, value.fragment, 4);
    put(raw.data() + 96, value.fragments, 4);
    put(raw.data() + 100, value.payload_bytes, 4);
    put(raw.data() + 104, value.total_bytes, 8);
    put(raw.data() + 112, value.sequence, 8);
    return raw;
}

header parse(const wire_header & raw) {
    if (!std::equal(std::begin(magic), std::end(magic), raw.begin()) || get(raw, 8, 2) != 1 || get(raw, 10, 2) != 0 ||
            get(raw, 12, 2) != wire_header_bytes || get(raw, 120, 8) != 0) {
        throw std::invalid_argument("unsupported GLM frame header");
    }
    header value;
    value.type = kind(get(raw, 14, 2));
    value.sender = uint32_t(get(raw, 16, 4));
    value.receiver = uint32_t(get(raw, 20, 4));
    value.epoch = get(raw, 24, 8);
    value.transaction = get(raw, 32, 8);
    value.chunk = get(raw, 40, 8);
    value.position = get(raw, 48, 8);
    std::copy(raw.begin() + 56, raw.begin() + 72, value.session.begin());
    value.generation = get(raw, 72, 8);
    value.rows = uint32_t(get(raw, 80, 4));
    value.hidden = uint32_t(get(raw, 84, 4));
    value.streams = uint16_t(get(raw, 88, 2));
    value.element = dtype(get(raw, 90, 2));
    value.fragment = uint32_t(get(raw, 92, 4));
    value.fragments = uint32_t(get(raw, 96, 4));
    value.payload_bytes = uint32_t(get(raw, 100, 4));
    value.total_bytes = get(raw, 104, 8);
    value.sequence = get(raw, 112, 8);
    return value;
}

digest authenticate_frame(const secret & key, const wire_header & raw, bytes_view payload) {
    return hmac_sha256(key, {{frame_domain, sizeof(frame_domain)}, {raw.data(), wire_mac_offset}, payload});
}

} // namespace

void limits::validate() const {
    if (!hidden || hidden > 4096 || streams != 4 || !rows || rows > 2048 || !context || context > 1048576 ||
            frame_bytes < (64u << 10) || frame_bytes > max_frame_bytes || frame_bytes % sizeof(float) != 0) {
        throw std::invalid_argument("invalid GLM negotiated resource limits");
    }
}

secret derive_link_key(const secret & psk, const binding & peer) {
    validate_binding(peer);
    if (zero(psk)) { throw std::invalid_argument("GLM authentication secret must be configured"); }
    uint8_t coordinates[9];
    put(coordinates, peer.sender, 4);
    put(coordinates + 4, peer.receiver, 4);
    coordinates[8] = uint8_t(peer.link);
    return hmac_sha256(psk, {{key_domain, sizeof(key_domain)}, {peer.server_nonce.data(), peer.server_nonce.size()},
        {peer.client_nonce.data(), peer.client_nonce.size()}, {peer.identity.data(), peer.identity.size()},
        {coordinates, sizeof(coordinates)}});
}

sender::sender(const secret & key, const binding & peer, const limits & bounds) : key(key), peer(peer), bounds(bounds) {
    validate_binding(peer);
    bounds.validate();
    if (zero(key)) { throw std::invalid_argument("GLM link key must be configured"); }
}

sender::~sender() { erase_secret(key); }

std::vector<uint8_t> sender::encode(header metadata, bytes_view payload) {
    if ((!payload.data && payload.size) || payload.size > max_frame_bytes) {
        throw std::invalid_argument("invalid or oversized GLM payload buffer");
    }
    metadata.payload_bytes = uint32_t(payload.size);
    metadata.sequence = next_sequence;
    validate(metadata, peer, bounds);
    auto raw = serialize(metadata);
    const auto tag = authenticate_frame(key, raw, payload);
    std::copy(tag.begin(), tag.end(), raw.begin() + wire_mac_offset);
    std::vector<uint8_t> result(wire_header_bytes + payload.size);
    std::copy(raw.begin(), raw.end(), result.begin());
    if (payload.size) { std::memcpy(result.data() + wire_header_bytes, payload.data, payload.size); }
    ++next_sequence;
    return result;
}

receiver::receiver(const secret & key, const binding & peer, const limits & bounds) : key(key), peer(peer), bounds(bounds) {
    validate_binding(peer);
    bounds.validate();
    if (zero(key)) { throw std::invalid_argument("GLM link key must be configured"); }
}

receiver::~receiver() { erase_secret(key); }

header receiver::inspect(const wire_header & raw) {
    if (failed) { throw std::runtime_error("GLM receiver is invalid; reconnect and authenticate again"); }
    try {
        const auto value = parse(raw);
        validate(value, peer, bounds);
        if (value.sequence != next_sequence) { throw std::invalid_argument("stale or out-of-order GLM frame"); }
        return value;
    } catch (...) {
        failed = true;
        throw;
    }
}

header receiver::authenticate(const wire_header & raw, bytes_view payload) {
    const auto value = inspect(raw);
    try {
        if ((!payload.data && payload.size) || value.payload_bytes != payload.size) {
            throw std::invalid_argument("truncated or inconsistent GLM payload");
        }
        digest supplied;
        std::copy(raw.begin() + wire_mac_offset, raw.end(), supplied.begin());
        const auto expected = authenticate_frame(key, raw, payload);
        if (!equal_digest(expected, supplied)) { throw std::invalid_argument("GLM frame authentication failed"); }
        ++next_sequence;
        return value;
    } catch (...) {
        failed = true;
        throw;
    }
}

} // namespace glm_rpc
