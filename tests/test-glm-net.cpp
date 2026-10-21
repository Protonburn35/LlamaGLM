#include "stage-net.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <future>
#include <functional>

using namespace glm_rpc;

static deadline shortly() { return std::chrono::steady_clock::now() + std::chrono::seconds(5); }

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void refused(const std::function<void()> & action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid network operation was accepted");
}

static void io_refused(const std::function<void()> & action, io_failure reason) {
    try { action(); } catch (const io_error & error) {
        require(error.reason == reason, "unexpected network failure category");
        return;
    }
    throw std::runtime_error("network operation did not fail as required");
}

struct fixture {
    secret psk;
    auth_identity identity;

    fixture() {
        psk.fill(0x23); // Public test key; never a runtime default.
        identity.identity.fill(0x67);
    }
};

static binding reverse(binding peer) {
    std::swap(peer.sender, peer.receiver);
    return peer;
}

static header boundary(uint32_t rows, uint32_t hidden, uint32_t fragments = 1) {
    header value;
    value.epoch = value.transaction = value.generation = 1;
    value.session[0] = 1;
    value.rows = rows;
    value.hidden = hidden;
    value.streams = 4;
    value.element = dtype::f32;
    value.total_bytes = uint64_t(rows) * hidden * 4 * 4;
    value.fragments = fragments;
    return value;
}

static void authenticated_exchange() {
    fixture f;
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    const auto endpoint = listener.local_endpoint();
    std::vector<uint8_t> payload(max_frame_bytes);
    for (size_t i = 0; i < payload.size(); ++i) { payload[i] = uint8_t(i); }
    auto server = std::async(std::launch::async, [&] {
        auto socket = listener.accept(shortly());
        const auto peer = authenticate_server(socket, f.psk, f.identity, shortly());
        receiver rx(derive_link_key(f.psk, peer), peer);
        const auto returning = reverse(peer);
        sender tx(derive_link_key(f.psk, returning), returning);
        for (uint32_t fragment = 0; fragment < 2; ++fragment) {
            const auto message = read_frame(socket, rx, shortly());
            require(message.metadata.fragment == fragment && message.metadata.fragments == 2, "network fragment coordinates changed");
            const size_t count = fragment == 0 ? max_frame_bytes : 65536;
            require(message.payload.size() == count && std::equal(message.payload.begin(), message.payload.end(), payload.begin()),
                    "network FP32 payload bits changed");
            auto reply = message.metadata;
            std::swap(reply.sender, reply.receiver);
            write_frame(socket, tx, reply, {message.payload.data(), message.payload.size()}, shortly());
        }
        const auto stats = socket.io_stats();
        const uint64_t framed = uint64_t(max_frame_bytes)+65536+2*wire_header_bytes;
        require(stats.read_bytes == framed+160 && stats.written_bytes == framed+320 && stats.recv_calls && stats.send_calls &&
                stats.read_waits >= stats.recv_calls && stats.write_waits >= stats.send_calls &&
                !stats.read_failures && !stats.write_failures && !stats.timeout_calls,
                "server stream telemetry differs from actual handshake/frame bytes");
    });
    auto socket = tcp_socket::connect(endpoint, shortly());
    const auto peer = authenticate_client(socket, f.psk, f.identity, shortly());
    sender tx(derive_link_key(f.psk, peer), peer);
    const auto returning = reverse(peer);
    receiver rx(derive_link_key(f.psk, returning), returning);
    auto metadata = boundary(65, 4096, 2);
    for (uint32_t fragment = 0; fragment < 2; ++fragment) {
        metadata.fragment = fragment;
        const size_t count = fragment == 0 ? max_frame_bytes : 65536;
        write_frame(socket, tx, metadata, {payload.data(), count}, shortly());
        const auto reply = read_frame(socket, rx, shortly());
        require(reply.payload.size() == count && std::equal(reply.payload.begin(), reply.payload.end(), payload.begin()),
                "authenticated network echo differs");
    }
    server.get();
    const auto stats = socket.io_stats();
    const uint64_t framed = uint64_t(max_frame_bytes)+65536+2*wire_header_bytes;
    require(stats.read_bytes == framed+320 && stats.written_bytes == framed+160 && stats.recv_calls && stats.send_calls &&
            stats.read_wait_ns && stats.write_wait_ns && !stats.read_failures && !stats.write_failures,
            "client stream telemetry differs from actual handshake/frame bytes or omitted readiness timing");
}

static void control_independence() {
    fixture f;
    auto control_identity = f.identity;
    control_identity.link = channel::control;
    auto data_listener = tcp_socket::listen({"127.0.0.1", 0});
    auto control_listener = tcp_socket::listen({"127.0.0.1", 0});
    auto server = std::async(std::launch::async, [&] {
        auto data = data_listener.accept(shortly());
        authenticate_server(data, f.psk, f.identity, shortly());
        auto control = control_listener.accept(shortly());
        const auto peer = authenticate_server(control, f.psk, control_identity, shortly());
        receiver rx(derive_link_key(f.psk, peer), peer);
        const auto message = read_frame(control, rx, shortly());
        require(message.metadata.type == kind::cancel, "control cancel was not received");
        const auto returning = reverse(peer);
        sender tx(derive_link_key(f.psk, returning), returning);
        header ack;
        ack.sender = 1; ack.receiver = 0; ack.type = kind::cancel_ack;
        write_frame(control, tx, ack, {nullptr, 0}, shortly());
        // No data payload was read; the separate control path must still complete.
        data.interrupt();
    });
    auto data = tcp_socket::connect(data_listener.local_endpoint(), shortly());
    const auto peer = authenticate_client(data, f.psk, f.identity, shortly());
    auto control = tcp_socket::connect(control_listener.local_endpoint(), shortly());
    const auto control_peer = authenticate_client(control, f.psk, control_identity, shortly());
    std::promise<void> started;
    auto began = started.get_future();
    auto writer = std::async(std::launch::async, [&] {
        started.set_value();
        try {
            limits bounds;
            bounds.hidden = 128;
            sender tx(derive_link_key(f.psk, peer), peer, bounds);
            std::vector<uint8_t> payload(max_frame_bytes, 0x5a);
            for (int i = 0; i < 16; ++i) { write_frame(data, tx, boundary(2048, 128), {payload.data(), payload.size()}, shortly()); }
            return false;
        } catch (const io_error &) { return true; }
    });
    began.get();
    require(writer.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready, "unread data stream did not apply backpressure");
    sender tx(derive_link_key(f.psk, control_peer), control_peer);
    const auto returning = reverse(control_peer);
    receiver rx(derive_link_key(f.psk, returning), returning);
    header cancel;
    cancel.type = kind::cancel;
    write_frame(control, tx, cancel, {nullptr, 0}, shortly());
    require(read_frame(control, rx, shortly()).metadata.type == kind::cancel_ack, "control path stalled behind activation data");
    data.interrupt();
    require(writer.get(), "cancelled data stream completed instead of refusing");
    server.get();
}

static void deadline_shutdown() {
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto client = tcp_socket::connect(listener.local_endpoint(), shortly());
    auto server = listener.accept(shortly());
    uint8_t byte;
    io_refused([&] { server.read_exact(&byte, 1, std::chrono::steady_clock::now() + std::chrono::milliseconds(30)); }, io_failure::timeout);
    io_refused([&] { server.read_exact(&byte, 1, shortly()); }, io_failure::closed);
    io_refused([&] { server.write_all({nullptr, 0}, shortly()); }, io_failure::closed);
    io_refused([&] { client.read_exact(&byte, 1, shortly()); }, io_failure::closed);
    const auto stats = server.io_stats();
    require(!stats.read_bytes && !stats.written_bytes && !stats.recv_calls && !stats.send_calls &&
            stats.read_failures == 2 && stats.write_failures == 1 && stats.timeout_calls == 1 && stats.read_waits == 1,
            "deadline/poisoned-call telemetry invented transfer bytes or lost failures");
}

static void partial_read_close() {
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto client = tcp_socket::connect(listener.local_endpoint(), shortly());
    auto server = listener.accept(shortly());
    const uint8_t partial[3] = {1, 2, 3};
    client.write_all({partial, sizeof(partial)}, shortly());
    client.interrupt();
    uint8_t output[5];
    io_refused([&] { server.read_exact(output, sizeof(output), shortly()); }, io_failure::closed);
    io_refused([&] { server.read_exact(output, 1, shortly()); }, io_failure::closed);
    const auto stats = server.io_stats();
    require(stats.read_bytes == 3 && client.io_stats().written_bytes == 3 && stats.read_failures == 2 &&
            !stats.timeout_calls && stats.recv_calls >= 1, "partial failed transfer telemetry counted requested bytes");
}

static void authentication_refusal() {
    fixture f;
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto server = std::async(std::launch::async, [&] {
        auto socket = listener.accept(shortly());
        try { authenticate_server(socket, f.psk, f.identity, shortly()); } catch (const std::exception &) { return true; }
        return false;
    });
    auto client = tcp_socket::connect(listener.local_endpoint(), shortly());
    auto wrong_key = f.psk;
    wrong_key[0] ^= 1;
    refused([&] { authenticate_client(client, wrong_key, f.identity, shortly()); });
    require(server.get(), "server admitted an unauthenticated connection");
}

static void forged_header_refusal() {
    fixture f;
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto server = std::async(std::launch::async, [&] {
        auto socket = listener.accept(shortly());
        const auto peer = authenticate_server(socket, f.psk, f.identity, shortly());
        limits bounds; bounds.hidden = 128;
        receiver rx(derive_link_key(f.psk, peer), peer, bounds);
        try { read_frame(socket, rx, shortly()); } catch (const std::invalid_argument &) { return true; }
        return false;
    });
    auto client = tcp_socket::connect(listener.local_endpoint(), shortly());
    const auto peer = authenticate_client(client, f.psk, f.identity, shortly());
    limits bounds; bounds.hidden = 128;
    sender tx(derive_link_key(f.psk, peer), peer, bounds);
    std::vector<uint8_t> payload(2048);
    auto packet = tx.encode(boundary(1, 128), {payload.data(), payload.size()});
    std::fill(packet.begin() + 100, packet.begin() + 104, 0xff);
    client.write_all({packet.data(), wire_header_bytes}, shortly());
    require(server.get(), "forged length was accepted or waited for a nonexistent body");
}

static void interface_refusal() {
    for (const auto * ip : {"0.0.0.0", "224.0.0.1", "255.255.255.255", "localhost", "invalid"}) {
        refused([&] { tcp_socket::listen({ip, 0}); });
    }
    refused([&] { tcp_socket::listen({"127.0.0.1", 0}, 9); });
    refused([&] { tcp_socket::connect({"127.0.0.1", 0}, shortly()); });
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    refused([&] { tcp_socket::listen(listener.local_endpoint()); });
    io_refused([&] { tcp_socket::connect(listener.local_endpoint(), std::chrono::steady_clock::now()); }, io_failure::timeout);
}

static void concurrent_interrupt() {
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto client = tcp_socket::connect(listener.local_endpoint(), shortly());
    auto server = listener.accept(shortly());
    auto read = std::async(std::launch::async, [&] {
        uint8_t byte;
        io_refused([&] { server.read_exact(&byte, 1, shortly()); }, io_failure::closed);
    });
    server.interrupt();
    read.get();
    auto accept = std::async(std::launch::async, [&] {
        io_refused([&] { listener.accept(shortly()); }, io_failure::closed);
    });
    listener.interrupt();
    accept.get();
}

static void concurrent_telemetry() {
    auto listener = tcp_socket::listen({"127.0.0.1",0});
    auto client = tcp_socket::connect(listener.local_endpoint(),shortly());
    auto server = listener.accept(shortly());
    std::vector<uint8_t> sent(max_frame_bytes,0x53), received(sent.size());
    auto read = std::async(std::launch::async,[&] { server.read_exact(received.data(),received.size(),shortly()); });
    auto write = std::async(std::launch::async,[&] { client.write_all({sent.data(),sent.size()},shortly()); });
    uint64_t previous = 0;
    while (read.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        const auto stats = server.io_stats();
        require(stats.read_bytes >= previous && stats.read_bytes <= sent.size(), "concurrent byte snapshot regressed or exceeded transfer");
        previous = stats.read_bytes;
    }
    write.get(); read.get();
    require(received == sent && server.io_stats().read_bytes == sent.size() && client.io_stats().written_bytes == sent.size(),
            "concurrent telemetry lost actual stream bytes");
    auto moved = std::move(client);
    require(!client.io_stats().written_bytes && moved.io_stats().written_bytes == sent.size(), "socket move lost or duplicated counters");
}

int main() {
    try {
        for (auto test : {authenticated_exchange, control_independence, deadline_shutdown, partial_read_close,
                          authentication_refusal, forged_header_refusal, interface_refusal, concurrent_interrupt, concurrent_telemetry}) { test(); }
        std::puts("GLM TCP: 9 suites passed; authenticated loopback, FP32 bits, control/backpressure, deadline, EOF, malformed peer, interrupt and actual-byte/concurrent telemetry");
        return 0;
    } catch (const io_error & error) {
        std::fprintf(stderr, "GLM TCP test failed: %s (category %d, native code %d)\n", error.what(), int(error.reason), error.code);
        return 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM TCP test failed: %s\n", error.what());
        return 1;
    }
}
