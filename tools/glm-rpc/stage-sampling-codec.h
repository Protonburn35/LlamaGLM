#pragma once

#include "stage-sampling.h"

namespace glm_rpc {

constexpr size_t sampling_setup_wire_bytes = 112, sampled_wire_bytes = 320;
using sampling_setup_wire = std::array<uint8_t,sampling_setup_wire_bytes>;
using sampled_wire = std::array<uint8_t,sampled_wire_bytes>;

sampling_setup_wire encode_sampling_setup(const sampling_config & expected, const epoch_stamp & stamp, const limits & bounds);
sampling_setup_wire encode_sampling_setup_ack(const sampling_config & expected, const epoch_stamp & stamp, const limits & bounds);
// Authenticate first. Require exact local configuration before BEGIN; a reconnect repeats
// agreement without resetting native sampling state. These ACKs are not resource admission.
void decode_sampling_setup(const header & authenticated, bytes_view payload, uint32_t local_rank,
        const sampling_config & expected, const epoch_stamp & stamp, const limits & bounds);
void decode_sampling_setup_ack(const header & authenticated, bytes_view payload, uint32_t expected_rank,
        const sampling_config & expected, const epoch_stamp & stamp, const limits & bounds);

// Caller must establish global completion first. Full vocabulary remains local to C.
sampled_wire encode_sampling_result(const sampling_result & result, const sampling_config & expected,
        const commit_certificate & committed, const limits & bounds);
sampling_result decode_sampling_result(const header & authenticated, bytes_view payload, const sampling_config & expected,
        const commit_certificate & committed, uint64_t expected_ordinal, const limits & bounds);

} // namespace glm_rpc
