#pragma once

#include "stage-wire.h"

namespace glm_rpc {

using auth_message = std::array<uint8_t, 160>;

struct auth_identity {
    digest identity{};
    uint32_t client = 0;
    uint32_t server = 1;
    channel link = channel::data;
};

// One handshake per connection. All messages have a fixed size; no peer-controlled allocations.
// Identity/limits must be agreed locally before use. A codec binding is available only after mutual proof.
class server_auth {
public:
    server_auth(const secret & psk, const auth_identity & expected);
    ~server_auth();
    server_auth(const server_auth &) = delete;
    server_auth & operator=(const server_auth &) = delete;

    auth_message challenge() const;
    auth_message accept_client(const auth_message & proof);
    const binding & agreed_binding() const;

private:
    secret psk;
    binding peer;
    bool complete = false;
    bool failed = false;
};

class client_auth {
public:
    client_auth(const secret & psk, const auth_identity & expected);
    ~client_auth();
    client_auth(const client_auth &) = delete;
    client_auth & operator=(const client_auth &) = delete;

    auth_message answer_server(const auth_message & challenge);
    void accept_server(const auth_message & proof);
    const binding & agreed_binding() const;

private:
    secret psk;
    binding peer;
    bool answered = false;
    bool complete = false;
    bool failed = false;
};

} // namespace glm_rpc
