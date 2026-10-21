#include "stage-auth.h"

#include <algorithm>
#include <stdexcept>

namespace glm_rpc {
namespace {

constexpr uint8_t magic[8] = {'L', 'L', 'G', 'L', 'M', 'H', 'S', '1'};
constexpr uint8_t auth_domain[] = "LlamaGLM RPC handshake v1";

bool zero(const digest & value) {
    return std::all_of(value.begin(), value.end(), [](uint8_t byte) { return byte == 0; });
}

binding configure(const secret & psk, const auth_identity & expected) {
    if (zero(psk) || zero(expected.identity) || expected.client > 2 || expected.server > 2 || expected.client == expected.server ||
            (expected.link != channel::data && expected.link != channel::control)) {
        throw std::invalid_argument("invalid GLM authentication configuration");
    }
    binding result;
    result.identity = expected.identity;
    result.sender = expected.client;
    result.receiver = expected.server;
    result.link = expected.link;
    return result;
}

void put(uint8_t * output, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) { output[i] = uint8_t(value >> (8 * i)); }
}

uint32_t get(const auth_message & message, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) { value |= uint32_t(message[offset + i]) << (8 * i); }
    return value;
}

digest tag(const secret & psk, const auth_message & message) {
    return hmac_sha256(psk, {{auth_domain, sizeof(auth_domain)}, {message.data(), wire_mac_offset}});
}

auth_message make_message(const secret & psk, const binding & peer, uint8_t phase) {
    auth_message result{};
    std::copy(std::begin(magic), std::end(magic), result.begin());
    result[8] = 1;
    result[12] = phase;
    result[13] = uint8_t(peer.link);
    put(result.data() + 16, phase == 2 ? peer.sender : peer.receiver);
    put(result.data() + 20, phase == 2 ? peer.receiver : peer.sender);
    std::copy(peer.server_nonce.begin(), peer.server_nonce.end(), result.begin() + 24);
    std::copy(peer.client_nonce.begin(), peer.client_nonce.end(), result.begin() + 56);
    std::copy(peer.identity.begin(), peer.identity.end(), result.begin() + 88);
    const auto proof = tag(psk, result);
    std::copy(proof.begin(), proof.end(), result.begin() + wire_mac_offset);
    return result;
}

binding verify_message(const secret & psk, const binding & peer, const auth_message & message, uint8_t phase) {
    if (!std::equal(std::begin(magic), std::end(magic), message.begin()) ||
            message[8] != 1 || message[9] || message[10] || message[11] || message[12] != phase ||
            message[13] != uint8_t(peer.link) || message[14] || message[15] ||
            get(message, 16) != (phase == 2 ? peer.sender : peer.receiver) ||
            get(message, 20) != (phase == 2 ? peer.receiver : peer.sender) ||
            !std::all_of(message.begin() + 120, message.begin() + 128, [](uint8_t byte) { return byte == 0; })) {
        throw std::invalid_argument("invalid GLM authentication envelope");
    }
    binding received = peer;
    std::copy(message.begin() + 24, message.begin() + 56, received.server_nonce.begin());
    std::copy(message.begin() + 56, message.begin() + 88, received.client_nonce.begin());
    digest identity, supplied;
    std::copy(message.begin() + 88, message.begin() + 120, identity.begin());
    std::copy(message.begin() + 128, message.end(), supplied.begin());
    if (!equal_digest(tag(psk, message), supplied) || !equal_digest(identity, peer.identity) || zero(received.server_nonce) ||
            (phase == 1 ? !zero(received.client_nonce) : zero(received.client_nonce)) ||
            (phase != 1 && !equal_digest(received.server_nonce, peer.server_nonce)) ||
            (phase == 3 && !equal_digest(received.client_nonce, peer.client_nonce))) {
        throw std::invalid_argument("GLM peer proof or negotiated identity mismatch");
    }
    return received;
}

} // namespace

server_auth::server_auth(const secret & psk, const auth_identity & expected) : psk(psk), peer(configure(psk, expected)) {
    peer.server_nonce = random_nonce();
}

server_auth::~server_auth() { erase_secret(psk); }

auth_message server_auth::challenge() const {
    if (failed || complete) { throw std::runtime_error("GLM server handshake is not awaiting a peer"); }
    return make_message(psk, peer, 1);
}

auth_message server_auth::accept_client(const auth_message & proof) {
    try {
        if (failed || complete) { throw std::runtime_error("GLM server handshake cannot accept another proof"); }
        peer = verify_message(psk, peer, proof, 2);
        const auto response = make_message(psk, peer, 3);
        complete = true;
        erase_secret(psk);
        return response;
    } catch (...) {
        failed = true;
        erase_secret(psk);
        throw;
    }
}

const binding & server_auth::agreed_binding() const {
    if (!complete || failed) { throw std::runtime_error("GLM server peer is not authenticated"); }
    return peer;
}

client_auth::client_auth(const secret & psk, const auth_identity & expected) : psk(psk), peer(configure(psk, expected)) {}

client_auth::~client_auth() { erase_secret(psk); }

auth_message client_auth::answer_server(const auth_message & challenge) {
    try {
        if (failed || answered || complete) { throw std::runtime_error("GLM client handshake cannot accept another challenge"); }
        peer = verify_message(psk, peer, challenge, 1);
        peer.client_nonce = random_nonce();
        const auto response = make_message(psk, peer, 2);
        answered = true;
        return response;
    } catch (...) {
        failed = true;
        erase_secret(psk);
        throw;
    }
}

void client_auth::accept_server(const auth_message & proof) {
    try {
        if (failed || !answered || complete) { throw std::runtime_error("GLM client handshake is not awaiting server proof"); }
        verify_message(psk, peer, proof, 3);
        complete = true;
        erase_secret(psk);
    } catch (...) {
        failed = true;
        erase_secret(psk);
        throw;
    }
}

const binding & client_auth::agreed_binding() const {
    if (!complete || failed) { throw std::runtime_error("GLM client peer is not authenticated"); }
    return peer;
}

} // namespace glm_rpc
