#pragma once

#include "stage-session.h"

namespace glm_rpc {

constexpr size_t receipt_wire_bytes = 224;
constexpr size_t commit_wire_bytes = 384;
using receipt_wire = std::array<uint8_t, receipt_wire_bytes>;
using commit_wire = std::array<uint8_t, commit_wire_bytes>;

receipt_wire encode_applied_receipt(const owner_receipt & receipt, const limits & bounds);
commit_wire encode_commit_certificate(const commit_certificate & certificate, const limits & bounds);

// Authenticate the enclosing control frame first. Pass the authenticated peer rank and
// locally pending key; fixed lengths, reserved fields and semantic compatibility are checked.
owner_receipt decode_applied_receipt(bytes_view payload, uint32_t authenticated_rank, const step_key & expected, const limits & bounds);
commit_certificate decode_commit_certificate(bytes_view payload, const step_key & expected, const limits & bounds);

// Correlate an already authenticated control envelope with its complete pending key/body.
// Control peers are A with B/C: applied ACKs go to A, decisions come from A. Data remains chained.
// Applied ACKs and commit decisions are distinct; global commit ACKs are not implemented.
owner_receipt decode_applied_ack(const header & authenticated, bytes_view payload, const step_key & expected, const limits & bounds);
commit_certificate decode_commit_decision(const header & authenticated, bytes_view payload, const step_key & expected, const limits & bounds);

} // namespace glm_rpc
