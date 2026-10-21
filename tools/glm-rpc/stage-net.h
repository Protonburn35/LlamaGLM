#pragma once

#include "stage-auth.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

namespace glm_rpc {

using deadline = std::chrono::steady_clock::time_point;

struct endpoint {
    std::string address; // Explicit numeric unicast IPv4 interface; wildcard addresses are refused.
    uint16_t port;
};

enum class io_failure { timeout, closed, system };

class io_error : public std::runtime_error {
public:
    io_error(io_failure reason, int code, const char * message);
    const io_failure reason;
    const int code;
};

// One reader and one writer per connection. interrupt() may run concurrently; join users before destruction/move.
// I/O failure shuts down the connection. Never resume a partially transferred frame on the same connection.
class tcp_socket {
public:
    tcp_socket() noexcept;
    ~tcp_socket();
    tcp_socket(tcp_socket &&) noexcept;
    tcp_socket & operator=(tcp_socket &&) noexcept;
    tcp_socket(const tcp_socket &) = delete;
    tcp_socket & operator=(const tcp_socket &) = delete;

    static tcp_socket listen(const endpoint & local, int backlog = 2);
    static tcp_socket connect(const endpoint & peer, deadline until);
    tcp_socket accept(deadline until);
    endpoint local_endpoint() const;

    void read_exact(uint8_t * output, size_t size, deadline until);
    void write_all(bytes_view input, deadline until);
    void interrupt() noexcept;

private:
    struct impl;
    explicit tcp_socket(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> state;
};

// Use one deadline for the entire handshake. No tensor payload allocation precedes mutual proof.
binding authenticate_client(tcp_socket & socket, const secret & psk, const auth_identity & expected, deadline until);
binding authenticate_server(tcp_socket & socket, const secret & psk, const auth_identity & expected, deadline until);

struct frame {
    header metadata;
    std::vector<uint8_t> payload;
};

frame read_frame(tcp_socket & socket, receiver & codec, deadline until);
void write_frame(tcp_socket & socket, sender & codec, header metadata, bytes_view payload, deadline until);

} // namespace glm_rpc
