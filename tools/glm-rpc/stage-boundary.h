#pragma once

#include "stage-net.h"

namespace glm_rpc {

// Synchronous whole-chunk bridge. Coordinates come from the local execution plan/transaction.
// Only forward A->B or B->C FP32 four-stream activations are accepted. One absolute deadline.
void send_boundary(tcp_socket & socket, sender & codec, header coordinates, const limits & bounds,
                   const float * values, size_t count, deadline until);

// Authenticates each fragment and checks all coordinates/order before exposing the complete input.
// input_budget bounds the one returned float buffer; account separately for one frame and model buffers.
// Any error discards partial input and permanently interrupts this connection. No transaction retry.
std::vector<float> receive_boundary(tcp_socket & socket, receiver & codec, header expected, const limits & bounds,
                                    uint64_t input_budget, deadline until);

} // namespace glm_rpc
