#pragma once

#include "stage-session.h"

namespace glm_rpc {

constexpr size_t receipt_wire_bytes = 224;
constexpr size_t commit_wire_bytes = 384;
constexpr size_t commit_ack_wire_bytes = 256;
constexpr size_t completion_wire_bytes = 176;
constexpr size_t terminal_wire_bytes = 272;
constexpr size_t begin_wire_bytes = 144;
using receipt_wire = std::array<uint8_t, receipt_wire_bytes>;
using commit_wire = std::array<uint8_t, commit_wire_bytes>;
using commit_ack_wire = std::array<uint8_t, commit_ack_wire_bytes>;
using completion_wire = std::array<uint8_t, completion_wire_bytes>;
using terminal_wire = std::array<uint8_t, terminal_wire_bytes>;
using begin_wire = std::array<uint8_t, begin_wire_bytes>;

// Locally expected state coordinates before A establishes this request's fingerprint.
struct begin_coordinates {
    epoch_stamp stamp;
    uint64_t transaction = 0, chunk = 0, position = 0;
    uint32_t rows = 0;
};

struct terminal_candidate {
    uint32_t token = 0;
    float logit = 0;
};

// Version 1 supports stateless greedy results only. Other sampling/RNG policies refuse.
// Raw FP32 logits are diagnostic top candidates, not normalized probabilities.
struct terminal_outcome {
    step_key step;
    digest certificate_digest{};
    uint32_t vocabulary = 0, selected_token = 0;
    float selected_logit = 0;
    uint32_t mode = 1, count = 0, row = 0;
    std::array<terminal_candidate,8> top{};
};

receipt_wire encode_applied_receipt(const owner_receipt & receipt, const limits & bounds);
commit_wire encode_commit_certificate(const commit_certificate & certificate, const limits & bounds);
digest fingerprint_commit(const commit_certificate & certificate, const limits & bounds);
// Call only with the receipt returned by successful local owner_transactions::commit.
commit_acknowledgement make_commit_ack(const owner_receipt & committed, const commit_certificate & certificate, const limits & bounds);
commit_ack_wire encode_commit_ack(const commit_acknowledgement & acknowledgement, const limits & bounds);
// Coordinator calls only with commit_barrier::committed(), after all three commit ACKs.
completion_wire encode_commit_completion(const commit_certificate & committed, const limits & bounds);
terminal_wire encode_terminal_outcome(const terminal_outcome & outcome, const limits & bounds);
begin_wire encode_step_begin(const step_key & step, const limits & bounds);

// Authenticate the enclosing control frame first. Pass the authenticated peer rank and
// locally pending key; fixed lengths, reserved fields and semantic compatibility are checked.
owner_receipt decode_applied_receipt(bytes_view payload, uint32_t authenticated_rank, const step_key & expected, const limits & bounds);
commit_certificate decode_commit_certificate(bytes_view payload, const step_key & expected, const limits & bounds);
commit_acknowledgement decode_commit_ack(bytes_view payload, uint32_t authenticated_rank, const step_key & expected,
        const digest & expected_certificate, const limits & bounds);

// Correlate an already authenticated control envelope with its complete pending key/body.
// Control peers are A with B/C: applied ACKs go to A, decisions come from A. Data remains chained.
// Applied ACKs, decisions and commit ACKs have distinct envelope/payload kinds.
owner_receipt decode_applied_ack(const header & authenticated, bytes_view payload, const step_key & expected, const limits & bounds);
commit_certificate decode_commit_decision(const header & authenticated, bytes_view payload, const step_key & expected, const limits & bounds);
commit_acknowledgement decode_committed_ack(const header & authenticated, bytes_view payload, const step_key & expected,
        const digest & expected_certificate, const limits & bounds);
// Owner calls only after its local commit; expected_certificate is that decision's fingerprint.
void decode_commit_completion(const header & authenticated, bytes_view payload, const step_key & expected,
        const digest & expected_certificate, const limits & bounds);
terminal_outcome decode_terminal_outcome(const header & authenticated, bytes_view payload, const step_key & expected,
        const digest & expected_certificate, uint32_t expected_vocabulary, uint32_t expected_candidates, const limits & bounds);
// Authenticate first. Only A introduces the request hash; all state coordinates remain local.
// A duplicate's returned full key must still match owner-ledger/cached history before reuse.
step_key decode_step_begin(const header & authenticated, bytes_view payload, const begin_coordinates & expected, const limits & bounds);

} // namespace glm_rpc
