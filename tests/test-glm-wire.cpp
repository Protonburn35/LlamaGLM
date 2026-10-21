#include "stage-wire.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace glm_rpc;

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void refused(const std::function<void()> & action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid protocol operation was accepted");
}

static std::string hex(bytes_view bytes) {
    const char digits[] = "0123456789abcdef";
    std::string result;
    for (size_t i = 0; i < bytes.size; ++i) {
        result += digits[bytes.data[i] >> 4];
        result += digits[bytes.data[i] & 15];
    }
    return result;
}

struct fixture {
    secret psk{};
    binding peer;
    limits bounds;
    header metadata;
    std::vector<uint8_t> payload;

    fixture() : payload(2048) {
        for (size_t i = 0; i < psk.size(); ++i) { psk[i] = uint8_t(i); }
        peer.server_nonce.fill(0x21);
        peer.client_nonce.fill(0x43);
        peer.identity.fill(0x65);
        bounds.hidden = 128;
        metadata.epoch = 0x0102030405060708ull;
        metadata.transaction = 9;
        metadata.chunk = 10;
        metadata.position = 5;
        for (size_t i = 0; i < metadata.session.size(); ++i) { metadata.session[i] = uint8_t(i + 1); }
        metadata.generation = 11;
        metadata.rows = 1;
        metadata.hidden = 128;
        metadata.streams = 4;
        metadata.element = dtype::f32;
        metadata.total_bytes = payload.size();
        for (size_t i = 0; i < payload.size(); ++i) { payload[i] = uint8_t(i); }
    }

    secret key() const { return derive_link_key(psk, peer); }
    bytes_view body() const { return {payload.data(), payload.size()}; }

    std::vector<uint8_t> packet() const {
        sender tx(key(), peer, bounds);
        return tx.encode(metadata, body());
    }
};

static wire_header prefix(const std::vector<uint8_t> & packet) {
    require(packet.size() >= wire_header_bytes, "fixture packet is truncated");
    wire_header raw;
    std::copy_n(packet.begin(), raw.size(), raw.begin());
    return raw;
}

static void crypto_vectors() {
    secret key{};
    std::fill_n(key.begin(), 20, 0x0b);
    const uint8_t message[] = "Hi There";
    const auto actual = hmac_sha256(key, {{message, 8}});
    require(hex({actual.data(), actual.size()}) == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            "RFC 4231 HMAC-SHA256 vector failed");
    require(equal_digest(actual, hmac_sha256(key, {{message, 3}, {message + 3, 5}})), "multipart HMAC differs");
    for (size_t i = 0; i < actual.size(); ++i) {
        auto changed = actual;
        changed[i] ^= 1;
        require(!equal_digest(actual, changed), "digest comparison missed a changed byte");
    }
    require(!equal_digest(random_nonce(), random_nonce()), "system nonces repeated");
    erase_secret(key);
    require(std::all_of(key.begin(), key.end(), [](uint8_t byte) { return byte == 0; }), "secret erase failed");
    refused([&] { hmac_sha256(key, {{nullptr, 1}}); });
}

static void binding_vectors() {
    fixture f;
    const auto key = f.key();
    require(hex({key.data(), key.size()}) == "c06a71b514aa89f4d4e642dcec3c1cd3efcc6f49f526415e75dd7ac3b8c90392",
            "independent Python link-key vector failed");
    for (int change = 0; change < 5; ++change) {
        auto peer = f.peer;
        if (change == 0) { peer.server_nonce[0] ^= 1; }
        if (change == 1) { peer.client_nonce[0] ^= 1; }
        if (change == 2) { peer.identity[0] ^= 1; }
        if (change == 3) { peer.link = channel::control; }
        if (change == 4) { std::swap(peer.sender, peer.receiver); }
        const auto other_key = derive_link_key(f.psk, peer);
        require(!equal_digest(key, other_key), "link binding was not separated");
        receiver wrong(other_key, peer, f.bounds);
        refused([&] { wrong.authenticate(prefix(f.packet()), f.body()); });
    }
    refused([&] { derive_link_key(secret{}, f.peer); });
    auto peer = f.peer;
    peer.server_nonce.fill(0);
    refused([&] { derive_link_key(f.psk, peer); });
    peer = f.peer;
    peer.receiver = peer.sender;
    refused([&] { derive_link_key(f.psk, peer); });
}

static void golden_frame() {
    fixture f;
    const auto packet = f.packet();
    auto raw = prefix(packet);
    // Generated independently with Python struct.pack and hmac.digest, not the C++ serializer.
    require(hex({raw.data(), raw.size()}) ==
        "4c4c474c4d52504301000000a0000600000000000100000008070605040302010900000000000000"
        "0a0000000000000005000000000000000102030405060708090a0b0c0d0e0f100b00000000000000"
        "01000000800000000400010000000000010000000008000000080000000000000100000000000000"
        "0000000000000000d103657dda9caa4aedabfcb6b6dd238442c684b09ad3b5fa5c7ca30274ecb4a3",
        "independent Python frame encoding/MAC vector failed");
    receiver rx(f.key(), f.peer, f.bounds);
    const auto peek = rx.inspect(raw);
    require(peek.payload_bytes == f.payload.size() && peek.sequence == 1, "fixed header inspection failed");
    const auto verified = rx.authenticate(raw, {packet.data() + wire_header_bytes, packet.size() - wire_header_bytes});
    require(verified.epoch == f.metadata.epoch && verified.position == 5 && verified.generation == 11, "coordinates changed");
    require(std::equal(f.payload.begin(), f.payload.end(), packet.begin() + wire_header_bytes), "FP32 payload bits changed");
}

static void corruption_refusal() {
    fixture f;
    const auto packet = f.packet();
    const auto original = prefix(packet);
    for (size_t byte = 0; byte < wire_header_bytes; ++byte) {
        auto raw = original;
        raw[byte] ^= 1;
        receiver rx(f.key(), f.peer, f.bounds);
        refused([&] { rx.authenticate(raw, f.body()); });
        refused([&] { rx.authenticate(original, f.body()); });
    }
    for (const size_t byte : {size_t(0), size_t(1), size_t(1023), f.payload.size() - 1}) {
        auto payload = f.payload;
        payload[byte] ^= 1;
        receiver rx(f.key(), f.peer, f.bounds);
        refused([&] { rx.authenticate(original, {payload.data(), payload.size()}); });
    }
    receiver truncated(f.key(), f.peer, f.bounds);
    refused([&] { truncated.authenticate(original, {f.payload.data(), f.payload.size() - 1}); });
}

static void sequence_refusal() {
    fixture f;
    sender tx(f.key(), f.peer, f.bounds);
    const auto first = prefix(tx.encode(f.metadata, f.body()));
    const auto second = prefix(tx.encode(f.metadata, f.body()));
    receiver rx(f.key(), f.peer, f.bounds);
    require(rx.authenticate(first, f.body()).sequence == 1, "first frame was not accepted");
    require(rx.authenticate(second, f.body()).sequence == 2, "second frame was not accepted");
    refused([&] { rx.authenticate(first, f.body()); });
    receiver skipped(f.key(), f.peer, f.bounds);
    refused([&] { skipped.inspect(second); });
    auto reconnected = f.peer;
    reconnected.server_nonce[0] ^= 1;
    receiver fresh(derive_link_key(f.psk, reconnected), reconnected, f.bounds);
    refused([&] { fresh.authenticate(first, f.body()); });
}

static void shape_refusal() {
    fixture f;
    sender tx(f.key(), f.peer, f.bounds);
    for (int change = 0; change < 11; ++change) {
        auto invalid = f.metadata;
        if (change == 0) { invalid.streams = 1; }
        if (change == 1) { invalid.hidden = 4096; }
        if (change == 2) { invalid.rows = 0; }
        if (change == 3) { invalid.rows = 2049; }
        if (change == 4) { invalid.total_bytes = std::numeric_limits<uint64_t>::max(); }
        if (change == 5) { invalid.position = std::numeric_limits<uint64_t>::max(); }
        if (change == 6) { invalid.element = dtype(2); }
        if (change == 7) { invalid.fragment = 1; }
        if (change == 8) { invalid.fragments = 0; }
        if (change == 9) { invalid.session.fill(0); }
        if (change == 10) { invalid.type = kind(65535); }
        refused([&] { tx.encode(invalid, f.body()); });
    }
    receiver rx(f.key(), f.peer, f.bounds);
    require(rx.authenticate(prefix(tx.encode(f.metadata, f.body())), f.body()).sequence == 1,
            "refused send advanced sequence state");
    auto invalid = f.bounds;
    invalid.frame_bytes = max_frame_bytes + 4;
    refused([&] { sender rejected(f.key(), f.peer, invalid); });
    invalid = f.bounds;
    invalid.context = 1048577;
    refused([&] { sender rejected(f.key(), f.peer, invalid); });
}

static void preallocation_refusal() {
    fixture f;
    const auto original = prefix(f.packet());
    for (const auto & field : {std::pair<size_t, size_t>{80, 4}, {92, 4}, {96, 4}, {100, 4}, {104, 8}, {112, 8}}) {
        auto raw = original;
        std::fill_n(raw.begin() + field.first, field.second, 0xff);
        receiver rx(f.key(), f.peer, f.bounds);
        // No payload buffer is allocated for these forged counts, lengths or sequences.
        refused([&] { rx.inspect(raw); });
        refused([&] { rx.inspect(original); });
    }
}

static void bounded_fragments() {
    fixture f;
    limits bounds;
    sender tx(f.key(), f.peer, bounds);
    receiver rx(f.key(), f.peer, bounds);
    auto metadata = f.metadata;
    metadata.hidden = 4096;
    metadata.rows = 2048;
    metadata.position = 0;
    metadata.total_bytes = max_chunk_bytes;
    metadata.fragments = 32;
    // Process a 128 MiB logical boundary while retaining only one 4 MiB payload and packet.
    std::vector<uint8_t> payload(max_frame_bytes, 0x5a);
    for (uint32_t fragment = 0; fragment < metadata.fragments; ++fragment) {
        metadata.fragment = fragment;
        const auto packet = tx.encode(metadata, {payload.data(), payload.size()});
        require(packet.size() == wire_header_bytes + max_frame_bytes, "fragment size exceeded the frame cap");
        const auto verified = rx.authenticate(prefix(packet), {packet.data() + wire_header_bytes, max_frame_bytes});
        require(verified.fragment == fragment && verified.total_bytes == max_chunk_bytes, "fragment metadata changed");
    }
    metadata.rows = 65;
    metadata.total_bytes = uint64_t(65) * 65536;
    metadata.fragments = 2;
    metadata.fragment = 1;
    const auto packet = tx.encode(metadata, {payload.data(), 65536});
    require(rx.authenticate(prefix(packet), {packet.data() + wire_header_bytes, 65536}).payload_bytes == 65536,
            "short final fragment failed");
}

static void control_isolation() {
    fixture f;
    auto peer = f.peer;
    peer.link = channel::control;
    const auto key = derive_link_key(f.psk, peer);
    sender tx(key, peer);
    receiver rx(key, peer);
    header cancel;
    cancel.type = kind::cancel;
    cancel.total_bytes = max_control_bytes;
    std::vector<uint8_t> payload(max_control_bytes + 1, 0);
    const auto packet = tx.encode(cancel, {payload.data(), max_control_bytes});
    require(rx.authenticate(prefix(packet), {packet.data() + wire_header_bytes, max_control_bytes}).type == kind::cancel,
            "bounded control frame failed");
    cancel.total_bytes = payload.size();
    refused([&] { tx.encode(cancel, {payload.data(), payload.size()}); });
    sender data(f.key(), f.peer, f.bounds);
    refused([&] { data.encode(cancel, {nullptr, 0}); });
    refused([&] { tx.encode(f.metadata, f.body()); });
}

int main() {
    try {
        for (auto test : {crypto_vectors, binding_vectors, golden_frame, corruption_refusal,
                          sequence_refusal, shape_refusal, preallocation_refusal, bounded_fragments, control_isolation}) {
            test();
        }
        std::puts("GLM wire: 9 suites passed; crypto vectors, framing, integrity, replay, limits and control isolation");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM wire test failed: %s\n", error.what());
        return 1;
    }
}
