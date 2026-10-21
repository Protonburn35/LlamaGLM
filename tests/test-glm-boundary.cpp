#include "stage-boundary.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>

using namespace glm_rpc;

static deadline shortly() { return std::chrono::steady_clock::now() + std::chrono::seconds(20); }
static void require(bool condition, const char * message) { if (!condition) { throw std::runtime_error(message); } }

struct fixture {
    secret psk{};
    auth_identity identity;
    limits bounds;
    header expected;
    fixture() {
        psk.fill(0x23); identity.identity.fill(0x67); // Public component fixture, never runtime defaults.
        bounds.hidden = 128; bounds.rows = 64; bounds.frame_bytes = 64u << 10;
        expected.epoch = expected.transaction = expected.generation = 1;
        expected.session[0] = 1; expected.position = 7;
        expected.rows = 33; expected.hidden = bounds.hidden; expected.streams = 4; expected.element = dtype::f32;
        update();
    }
    void update() {
        expected.total_bytes = uint64_t(expected.rows) * expected.hidden * 4 * sizeof(float);
        expected.fragments = uint32_t((expected.total_bytes + bounds.frame_bytes - 1) / bounds.frame_bytes);
    }
};

static std::vector<float> pattern(size_t count) {
    std::vector<float> values(count);
    const std::array<uint32_t, 8> special{0, 0x80000000, 0x7fc12345, 0x7f800000, 0xff800000, 1, 0x007fffff, 0x3f800001};
    for (size_t i = 0; i < count; ++i) {
        const uint32_t bits = i < special.size() ? special[i] : 0x3f000000u | uint32_t(i & 0x7fffff);
        std::memcpy(values.data() + i, &bits, sizeof(bits));
    }
    return values;
}

static void bit_exact(uint32_t rows, uint32_t hidden, uint32_t frame_bytes, bool canonical = false) {
    fixture f;
    f.bounds.hidden = f.expected.hidden = hidden; f.bounds.rows = f.expected.rows = rows;
    f.bounds.frame_bytes = frame_bytes;
    f.expected.position = f.bounds.context - rows;
    f.update();
    const auto values = pattern(size_t(f.expected.total_bytes / sizeof(float)));
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto server = std::async(std::launch::async, [&] {
        auto socket = listener.accept(shortly());
        const auto peer = authenticate_server(socket, f.psk, f.identity, shortly());
        receiver rx(derive_link_key(f.psk, peer), peer, f.bounds);
        const auto input = receive_boundary(socket, rx, f.expected, f.bounds, f.expected.total_bytes, shortly());
        require(input.size() == values.size() && std::memcmp(input.data(), values.data(), f.expected.total_bytes) == 0,
                "fragment bridge changed IEEE FP32 bits or final positions");
    });
    auto socket = tcp_socket::connect(listener.local_endpoint(), shortly());
    const auto peer = authenticate_client(socket, f.psk, f.identity, shortly());
    sender tx(derive_link_key(f.psk, peer), peer, f.bounds);
    if (canonical) {
        std::vector<uint8_t> encoded(values.size()*sizeof(float));
        for (size_t i = 0; i < values.size(); ++i) {
            uint32_t bits; std::memcpy(&bits,values.data()+i,sizeof(bits));
            for (size_t byte = 0; byte < 4; ++byte) { encoded[i*4+byte] = uint8_t(bits >> (byte*8)); }
        }
        send_boundary_bytes(socket, tx, f.expected, f.bounds, {encoded.data(),encoded.size()}, shortly());
    } else {
        send_boundary(socket, tx, f.expected, f.bounds, values.data(), values.size(), shortly());
    }
    server.get();
}

static void completion_barrier() {
    fixture f;
    const auto values = pattern(size_t(f.expected.total_bytes / sizeof(float)));
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto server = std::async(std::launch::async, [&] {
        auto socket = listener.accept(shortly());
        const auto peer = authenticate_server(socket, f.psk, f.identity, shortly());
        receiver rx(derive_link_key(f.psk, peer), peer, f.bounds);
        return receive_boundary(socket, rx, f.expected, f.bounds, f.expected.total_bytes, shortly());
    });
    auto socket = tcp_socket::connect(listener.local_endpoint(), shortly());
    const auto peer = authenticate_client(socket, f.psk, f.identity, shortly());
    sender tx(derive_link_key(f.psk, peer), peer, f.bounds);
    write_frame(socket, tx, f.expected, {reinterpret_cast<const uint8_t *>(values.data()), f.bounds.frame_bytes}, shortly());
    require(server.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready, "partial boundary escaped to the caller");
    auto last = f.expected; last.fragment = 1;
    write_frame(socket, tx, last, {reinterpret_cast<const uint8_t *>(values.data()) + f.bounds.frame_bytes,
        size_t(last.total_bytes - f.bounds.frame_bytes)}, shortly());
    const auto input = server.get();
    require(input.size() == values.size() && std::memcmp(input.data(), values.data(), f.expected.total_bytes) == 0,
            "completed fragmented input differs");
}

static void coordinate_refusal() {
    const std::vector<std::function<void(header &)>> changes{
        [](auto & h) { ++h.epoch; }, [](auto & h) { ++h.transaction; }, [](auto & h) { ++h.chunk; },
        [](auto & h) { ++h.position; }, [](auto & h) { ++h.generation; }, [](auto & h) { ++h.session[0]; },
        [](auto & h) { ++h.rows; h.total_bytes += 128 * 4 * sizeof(float); },
        [](auto & h) { h.fragment = 0; },
    };
    for (const auto & change : changes) {
        fixture f;
        auto listener = tcp_socket::listen({"127.0.0.1", 0});
        auto server = std::async(std::launch::async, [&] {
            auto socket = listener.accept(shortly());
            const auto peer = authenticate_server(socket, f.psk, f.identity, shortly());
            receiver rx(derive_link_key(f.psk, peer), peer, f.bounds);
            try { receive_boundary(socket, rx, f.expected, f.bounds, f.expected.total_bytes, shortly()); }
            catch (const std::invalid_argument & e) {
                require(std::strstr(e.what(), "active execution plan") != nullptr, "coordinate test failed for an unrelated reason");
                uint8_t byte;
                try { socket.read_exact(&byte, 1, shortly()); } catch (const io_error &) { return; }
                throw std::runtime_error("invalid chunk connection was reusable");
            }
            throw std::runtime_error("mixed or repeated fragment coordinates were accepted");
        });
        auto socket = tcp_socket::connect(listener.local_endpoint(), shortly());
        const auto peer = authenticate_client(socket, f.psk, f.identity, shortly());
        sender tx(derive_link_key(f.psk, peer), peer, f.bounds);
        std::vector<uint8_t> payload(f.bounds.frame_bytes, 0x5a);
        write_frame(socket, tx, f.expected, {payload.data(), payload.size()}, shortly());
        auto last = f.expected; last.fragment = 1; change(last);
        const auto size = size_t(std::min<uint64_t>(f.bounds.frame_bytes, last.total_bytes - uint64_t(last.fragment) * f.bounds.frame_bytes));
        write_frame(socket, tx, last, {payload.data(), size}, shortly());
        server.get();
    }
}

static void broken_input(bool corrupt) {
    fixture f;
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto server = std::async(std::launch::async, [&] {
        auto socket = listener.accept(shortly());
        const auto peer = authenticate_server(socket, f.psk, f.identity, shortly());
        receiver rx(derive_link_key(f.psk, peer), peer, f.bounds);
        try { receive_boundary(socket, rx, f.expected, f.bounds, f.expected.total_bytes, shortly()); }
        catch (const io_error & error) { require(!corrupt && error.reason == io_failure::closed, "wrong truncation failure"); return; }
        catch (const std::invalid_argument & error) {
            require(corrupt && std::strstr(error.what(), "authentication failed"), "wrong corrupt-fragment failure"); return;
        }
        throw std::runtime_error("broken complete boundary was returned");
    });
    auto socket = tcp_socket::connect(listener.local_endpoint(), shortly());
    const auto peer = authenticate_client(socket, f.psk, f.identity, shortly());
    sender tx(derive_link_key(f.psk, peer), peer, f.bounds);
    std::vector<uint8_t> payload(f.bounds.frame_bytes, 0x5a);
    write_frame(socket, tx, f.expected, {payload.data(), payload.size()}, shortly());
    auto last = f.expected; last.fragment = 1;
    auto packet = tx.encode(last, {payload.data(), size_t(last.total_bytes - f.bounds.frame_bytes)});
    if (corrupt) { packet.back() ^= 1; }
    socket.write_all({packet.data(), corrupt ? packet.size() : packet.size() - 1}, shortly());
    if (!corrupt) { socket.interrupt(); }
    server.get();
}

static void reserved_budget() {
    fixture f;
    for (const uint64_t budget : {uint64_t(0), f.expected.total_bytes - 1, max_chunk_bytes + 1}) {
        auto listener = tcp_socket::listen({"127.0.0.1", 0});
        auto socket = tcp_socket::connect(listener.local_endpoint(), shortly());
        auto peer_socket = listener.accept(shortly());
        binding peer; peer.server_nonce.fill(1); peer.client_nonce.fill(2); peer.identity.fill(3);
        receiver rx(derive_link_key(f.psk, peer), peer, f.bounds);
        try { receive_boundary(peer_socket, rx, f.expected, f.bounds, budget, shortly()); }
        catch (const std::invalid_argument & e) {
            require(std::strstr(e.what(), "local reserved budget"), "wrong local budget failure"); continue;
        }
        throw std::runtime_error("invalid local boundary budget was accepted");
    }
}

static void chunk_deadline() {
    fixture f;
    auto listener = tcp_socket::listen({"127.0.0.1", 0});
    auto server = std::async(std::launch::async, [&] {
        auto socket = listener.accept(shortly());
        const auto peer = authenticate_server(socket, f.psk, f.identity, shortly());
        receiver rx(derive_link_key(f.psk, peer), peer, f.bounds);
        try { receive_boundary(socket, rx, f.expected, f.bounds, f.expected.total_bytes,
            std::chrono::steady_clock::now() + std::chrono::milliseconds(50)); }
        catch (const io_error & error) { require(error.reason == io_failure::timeout, "chunk failed before its deadline"); return; }
        throw std::runtime_error("incomplete chunk escaped its deadline");
    });
    auto socket = tcp_socket::connect(listener.local_endpoint(), shortly());
    const auto peer = authenticate_client(socket, f.psk, f.identity, shortly());
    sender tx(derive_link_key(f.psk, peer), peer, f.bounds);
    std::vector<uint8_t> payload(f.bounds.frame_bytes, 0x5a);
    write_frame(socket, tx, f.expected, {payload.data(), payload.size()}, shortly());
    server.get();
}

static void canonical_size_refusal() {
    fixture f;
    for (size_t size : {size_t(0),size_t(f.expected.total_bytes-1),size_t(f.expected.total_bytes+1)}) {
        auto listener = tcp_socket::listen({"127.0.0.1",0});
        auto socket = tcp_socket::connect(listener.local_endpoint(),shortly());
        auto peer_socket = listener.accept(shortly());
        binding peer; peer.server_nonce.fill(1); peer.client_nonce.fill(2); peer.identity.fill(3);
        sender tx(derive_link_key(f.psk,peer),peer,f.bounds);
        uint8_t byte = 0;
        try { send_boundary_bytes(socket,tx,f.expected,f.bounds,{&byte,size},shortly()); }
        catch (const std::invalid_argument &) {
            try { socket.write_all({&byte,1},shortly()); }
            catch (const io_error &) { continue; }
            throw std::runtime_error("invalid canonical payload left connection reusable");
        }
        throw std::runtime_error("incorrect canonical boundary size was accepted");
    }
}

int main() {
    try {
        bit_exact(2048, 4096, max_frame_bytes); bit_exact(33, 128, 64u << 10);
        bit_exact(33,128,64u<<10,true); canonical_size_refusal();
        completion_barrier(); coordinate_refusal(); broken_input(false); broken_input(true); reserved_budget(); chunk_deadline();
        std::puts("GLM boundary: ten suites passed (128 MiB, tails, canonical cached bytes/size refusal, complete barrier, coordinates, truncation, MAC, budget, deadline)");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM boundary test failed: %s\n", error.what());
        return 1;
    }
}
