#include "stage-net.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace glm_rpc {
namespace {

#ifdef _WIN32
using native_socket = SOCKET;
constexpr native_socket invalid_socket = INVALID_SOCKET;
using socket_size = int;

void initialize_network() {
    struct runtime {
        runtime() {
            WSADATA data;
            const int status = WSAStartup(MAKEWORD(2, 2), &data);
            if (status) { throw io_error(io_failure::system, status, "cannot initialize Windows networking"); }
        }
        ~runtime() { WSACleanup(); }
    };
    static runtime network;
}

int last_error() { return WSAGetLastError(); }
bool pending(int code) { return code == WSAEWOULDBLOCK || code == WSAEINPROGRESS; }
bool interrupted(int code) { return code == WSAEINTR; }
void close_socket(native_socket socket) { closesocket(socket); }
void shutdown_socket(native_socket socket) { shutdown(socket, SD_BOTH); }
#else
using native_socket = int;
constexpr native_socket invalid_socket = -1;
using socket_size = socklen_t;

void initialize_network() {}
int last_error() { return errno; }
bool pending(int code) { return code == EWOULDBLOCK || code == EAGAIN || code == EINPROGRESS; }
bool interrupted(int code) { return code == EINTR; }
void close_socket(native_socket socket) { close(socket); }
void shutdown_socket(native_socket socket) { shutdown(socket, SHUT_RDWR); }
#endif

io_failure failure_reason(int code) {
#ifdef _WIN32
    if (code == WSAECONNRESET || code == WSAECONNABORTED || code == WSAENOTCONN || code == WSAESHUTDOWN) {
        return io_failure::closed;
    }
    if (code == WSAETIMEDOUT) { return io_failure::timeout; }
#else
    if (code == ECONNRESET || code == ECONNABORTED || code == ENOTCONN || code == EPIPE) {
        return io_failure::closed;
    }
    if (code == ETIMEDOUT) { return io_failure::timeout; }
#endif
    return io_failure::system;
}

void check_deadline(deadline until) {
    if (std::chrono::steady_clock::now() >= until) { throw io_error(io_failure::timeout, 0, "GLM connection deadline exceeded"); }
}

sockaddr_in address(const endpoint & value, bool allow_ephemeral) {
    sockaddr_in result{};
    result.sin_family = AF_INET;
    result.sin_port = htons(value.port);
    if ((!allow_ephemeral && !value.port) || value.address.find('\0') != std::string::npos ||
            inet_pton(AF_INET, value.address.c_str(), &result.sin_addr) != 1) {
        throw std::invalid_argument("GLM TCP endpoint requires a numeric IPv4 address and valid port");
    }
    const auto host = ntohl(result.sin_addr.s_addr);
    if ((host >> 24) == 0 || (host >> 24) >= 224) {
        throw std::invalid_argument("GLM TCP requires an explicit unicast interface");
    }
    return result;
}

void nonblocking(native_socket socket) {
#ifdef _WIN32
    u_long enabled = 1;
    if (!SetHandleInformation(reinterpret_cast<HANDLE>(socket), HANDLE_FLAG_INHERIT, 0)) {
        throw io_error(io_failure::system, int(GetLastError()), "cannot prevent GLM socket inheritance");
    }
    if (ioctlsocket(socket, FIONBIO, &enabled) != 0) {
#else
    const int flags = fcntl(socket, F_GETFL, 0);
    const int descriptor_flags = fcntl(socket, F_GETFD, 0);
    if (flags < 0 || descriptor_flags < 0 || fcntl(socket, F_SETFL, flags | O_NONBLOCK) != 0 ||
            fcntl(socket, F_SETFD, descriptor_flags | FD_CLOEXEC) != 0) {
#endif
        throw io_error(io_failure::system, last_error(), "cannot configure nonblocking GLM socket");
    }
}

void stream_options(native_socket socket) {
    const int enabled = 1;
    const int buffer_bytes = 1 << 20;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&enabled), sizeof(enabled)) != 0 ||
            setsockopt(socket, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char *>(&buffer_bytes), sizeof(buffer_bytes)) != 0 ||
            setsockopt(socket, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char *>(&buffer_bytes), sizeof(buffer_bytes)) != 0) {
        throw io_error(io_failure::system, last_error(), "cannot configure bounded GLM stream buffers");
    }
}

void wait_socket(native_socket socket, bool writing, deadline until, const std::atomic<bool> & stopped) {
    while (true) {
        if (stopped.load()) { throw io_error(io_failure::closed, 0, "GLM connection was interrupted"); }
        check_deadline(until);
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(until - std::chrono::steady_clock::now()).count();
        const int milliseconds = int(std::max<int64_t>(1, std::min<int64_t>(remaining, 1000)));
#ifdef _WIN32
        WSAPOLLFD descriptor{socket, short(writing ? POLLWRNORM : POLLRDNORM), 0};
        const int result = WSAPoll(&descriptor, 1, milliseconds);
#else
        pollfd descriptor{socket, short(writing ? POLLOUT : POLLIN), 0};
        const int result = poll(&descriptor, 1, milliseconds);
#endif
        if (result > 0) {
            if (stopped.load()) { throw io_error(io_failure::closed, 0, "GLM connection was interrupted"); }
            if (descriptor.revents & POLLNVAL) { throw io_error(io_failure::closed, 0, "GLM socket is closed"); }
            return;
        }
        if (result < 0 && !interrupted(last_error())) {
            throw io_error(io_failure::system, last_error(), "cannot wait on GLM socket");
        }
    }
}

} // namespace

struct tcp_socket::impl {
    native_socket handle;
    bool listener;
    std::atomic<bool> stopped{false};

    impl(native_socket handle, bool listener) : handle(handle), listener(listener) {}
    ~impl() { if (handle != invalid_socket) { close_socket(handle); } }
};

io_error::io_error(io_failure reason, int code, const char * message) : std::runtime_error(message), reason(reason), code(code) {}
tcp_socket::tcp_socket() noexcept = default;
tcp_socket::tcp_socket(std::unique_ptr<impl> state) noexcept : state(std::move(state)) {}
tcp_socket::~tcp_socket() = default;
tcp_socket::tcp_socket(tcp_socket &&) noexcept = default;
tcp_socket & tcp_socket::operator=(tcp_socket &&) noexcept = default;

tcp_socket tcp_socket::listen(const endpoint & local, int backlog) {
    initialize_network();
    const auto local_address = address(local, true);
    if (backlog < 1 || backlog > 8) { throw std::invalid_argument("GLM TCP backlog exceeds connection limit"); }
    auto state = std::make_unique<impl>(invalid_socket, true);
    const auto handle = state->handle = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (handle == invalid_socket) { throw io_error(io_failure::system, last_error(), "cannot create GLM listener"); }
    nonblocking(handle);
#ifdef _WIN32
    const int exclusive = 1;
    if (setsockopt(handle, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char *>(&exclusive), sizeof(exclusive)) != 0) {
        throw io_error(io_failure::system, last_error(), "cannot reserve exclusive GLM interface");
    }
#endif
    if (bind(handle, reinterpret_cast<const sockaddr *>(&local_address), sizeof(local_address)) != 0 || ::listen(handle, backlog) != 0) {
        throw io_error(io_failure::system, last_error(), "cannot bind GLM listener to configured interface/port");
    }
    return tcp_socket(std::move(state));
}

tcp_socket tcp_socket::connect(const endpoint & peer, deadline until) {
    initialize_network();
    const auto peer_address = address(peer, false);
    check_deadline(until);
    auto state = std::make_unique<impl>(invalid_socket, false);
    const auto handle = state->handle = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (handle == invalid_socket) { throw io_error(io_failure::system, last_error(), "cannot create GLM connection"); }
    nonblocking(handle);
    if (::connect(handle, reinterpret_cast<const sockaddr *>(&peer_address), sizeof(peer_address)) != 0) {
        const int code = last_error();
        if (!pending(code)) { throw io_error(io_failure::system, code, "cannot connect GLM peer"); }
        wait_socket(handle, true, until, state->stopped);
        int error = 0;
        socket_size size = sizeof(error);
        if (getsockopt(handle, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&error), &size) != 0 || error != 0) {
            throw io_error(io_failure::system, error ? error : last_error(), "GLM peer connection failed");
        }
    }
    check_deadline(until);
    stream_options(handle);
    return tcp_socket(std::move(state));
}

tcp_socket tcp_socket::accept(deadline until) {
    if (!state || !state->listener) { throw std::invalid_argument("GLM socket is not a listener"); }
    auto connection = std::make_unique<impl>(invalid_socket, false);
    while (true) {
        wait_socket(state->handle, false, until, state->stopped);
        const auto handle = ::accept(state->handle, nullptr, nullptr);
        if (handle != invalid_socket) {
            connection->handle = handle;
            nonblocking(handle);
            stream_options(handle);
            if (state->stopped.load()) { throw io_error(io_failure::closed, 0, "GLM listener was interrupted"); }
            return tcp_socket(std::move(connection));
        }
        const int code = last_error();
        if (!pending(code) && !interrupted(code)) { throw io_error(io_failure::system, code, "cannot accept GLM connection"); }
    }
}

endpoint tcp_socket::local_endpoint() const {
    if (!state) { throw std::invalid_argument("GLM socket is empty"); }
    sockaddr_in local{};
    socket_size size = sizeof(local);
    char text[INET_ADDRSTRLEN];
    if (getsockname(state->handle, reinterpret_cast<sockaddr *>(&local), &size) != 0 ||
            !inet_ntop(AF_INET, &local.sin_addr, text, sizeof(text))) {
        throw io_error(io_failure::system, last_error(), "cannot inspect GLM local endpoint");
    }
    return {text, ntohs(local.sin_port)};
}

void tcp_socket::interrupt() noexcept {
    if (state && !state->stopped.exchange(true)) { shutdown_socket(state->handle); }
}

void tcp_socket::read_exact(uint8_t * output, size_t size, deadline until) {
    if (!state || state->listener || (!output && size)) { throw std::invalid_argument("invalid GLM stream read"); }
    try {
        if (state->stopped.load()) { throw io_error(io_failure::closed, 0, "GLM connection was interrupted"); }
        while (size) {
            wait_socket(state->handle, false, until, state->stopped);
            const int count = int(std::min<size_t>(size, INT_MAX));
            const auto received = recv(state->handle, reinterpret_cast<char *>(output), count, 0);
            if (received > 0) { output += received; size -= size_t(received); continue; }
            if (received == 0) { throw io_error(io_failure::closed, 0, "GLM peer closed during read"); }
            const int code = last_error();
            if (!pending(code) && !interrupted(code)) { throw io_error(failure_reason(code), code, "GLM stream read failed"); }
        }
        check_deadline(until);
        if (state->stopped.load()) { throw io_error(io_failure::closed, 0, "GLM connection was interrupted"); }
    } catch (...) { interrupt(); throw; }
}

void tcp_socket::write_all(bytes_view input, deadline until) {
    if (!state || state->listener || (!input.data && input.size)) { throw std::invalid_argument("invalid GLM stream write"); }
    try {
        if (state->stopped.load()) { throw io_error(io_failure::closed, 0, "GLM connection was interrupted"); }
        while (input.size) {
            wait_socket(state->handle, true, until, state->stopped);
            const int count = int(std::min<size_t>(input.size, INT_MAX));
#ifdef _WIN32
            const int flags = 0;
#else
            const int flags = MSG_NOSIGNAL;
#endif
            const auto sent = send(state->handle, reinterpret_cast<const char *>(input.data), count, flags);
            if (sent > 0) { input.data += sent; input.size -= size_t(sent); continue; }
            if (sent == 0) { throw io_error(io_failure::closed, 0, "GLM peer closed during write"); }
            const int code = last_error();
            if (!pending(code) && !interrupted(code)) { throw io_error(failure_reason(code), code, "GLM stream write failed"); }
        }
        check_deadline(until);
        if (state->stopped.load()) { throw io_error(io_failure::closed, 0, "GLM connection was interrupted"); }
    } catch (...) { interrupt(); throw; }
}

binding authenticate_client(tcp_socket & socket, const secret & psk, const auth_identity & expected, deadline until) {
    try {
        client_auth auth(psk, expected);
        auth_message message;
        socket.read_exact(message.data(), message.size(), until);
        const auto response = auth.answer_server(message);
        socket.write_all({response.data(), response.size()}, until);
        socket.read_exact(message.data(), message.size(), until);
        auth.accept_server(message);
        check_deadline(until);
        return auth.agreed_binding();
    } catch (...) { socket.interrupt(); throw; }
}

binding authenticate_server(tcp_socket & socket, const secret & psk, const auth_identity & expected, deadline until) {
    try {
        server_auth auth(psk, expected);
        const auto challenge = auth.challenge();
        socket.write_all({challenge.data(), challenge.size()}, until);
        auth_message message;
        socket.read_exact(message.data(), message.size(), until);
        const auto response = auth.accept_client(message);
        socket.write_all({response.data(), response.size()}, until);
        check_deadline(until);
        return auth.agreed_binding();
    } catch (...) { socket.interrupt(); throw; }
}

frame read_frame(tcp_socket & socket, receiver & codec, deadline until) {
    try {
        wire_header raw;
        socket.read_exact(raw.data(), raw.size(), until);
        const auto metadata = codec.inspect(raw);
        frame result{metadata, std::vector<uint8_t>(metadata.payload_bytes)};
        socket.read_exact(result.payload.data(), result.payload.size(), until);
        codec.authenticate(raw, {result.payload.data(), result.payload.size()});
        check_deadline(until);
        return result;
    } catch (...) { socket.interrupt(); throw; }
}

void write_frame(tcp_socket & socket, sender & codec, header metadata, bytes_view payload, deadline until) {
    try {
        const auto packet = codec.encode(metadata, payload);
        socket.write_all({packet.data(), packet.size()}, until);
    } catch (...) { socket.interrupt(); throw; }
}

} // namespace glm_rpc
