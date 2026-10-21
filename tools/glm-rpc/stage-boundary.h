#pragma once

#include "stage-net.h"

namespace glm_rpc {

// Synchronous whole-chunk bridge. Coordinates come from the local execution plan/transaction.
// Only forward A->B or B->C FP32 four-stream activations are accepted. One absolute deadline.
void send_boundary(tcp_socket & socket, sender & codec, header coordinates, const limits & bounds,
                   const float * values, size_t count, deadline until);

// Already canonical little-endian FP32 bytes, for immutable cached replies. No float aliasing
// or conversion: caller binds these exact unfragmented bytes to the applied output receipt.
void send_boundary_bytes(tcp_socket & socket, sender & codec, header coordinates, const limits & bounds,
                         bytes_view payload, deadline until);

// Authenticates each fragment and checks all coordinates/order before exposing the complete input.
// input_budget bounds the one returned float buffer; account separately for one frame and model buffers.
// Any error discards partial input and permanently interrupts this connection. No transaction retry.
std::vector<float> receive_boundary(tcp_socket & socket, receiver & codec, header expected, const limits & bounds,
                                    uint64_t input_budget, deadline until);

// Exclusive caller-owned float storage, reusable after synchronous consumption. Each
// fragment reads directly into its final slice; no frame/reassembly payload is allocated.
// Caller reserves/protects storage before peer work. Failure clears the entire active chunk.
size_t receive_boundary_into(tcp_socket & socket, receiver & codec, header expected, const limits & bounds,
                             float * storage, size_t capacity, deadline until);

} // namespace glm_rpc
