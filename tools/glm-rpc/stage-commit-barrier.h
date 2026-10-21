#pragma once

#include "stage-session-codec.h"

namespace glm_rpc {

enum class barrier_phase { collecting, decided, complete, invalid };

// One bounded single-writer coordinator transaction. This component checks already
// authenticated facts; it does not run models, store replies, send messages or reset peers.
// A is local rank 0. B/C facts require their directly authenticated control connections.
class commit_barrier {
public:
    commit_barrier(const step_key & expected, const limits & bounds);
    commit_barrier(const commit_barrier &) = delete;
    commit_barrier & operator=(const commit_barrier &) = delete;

    void record_applied(uint32_t authenticated_rank, const owner_receipt & receipt);
    const commit_certificate & decision() const;
    void record_committed(uint32_t authenticated_rank, const commit_acknowledgement & acknowledgement);
    // Available only after all three distinct owners committed this exact certificate.
    // Match retained C output to owners[2].output_digest before publishing any reply.
    const commit_certificate & committed() const;
    void invalidate() noexcept;

    barrier_phase phase() const { return state; }

private:
    step_key expected;
    limits bounds;
    barrier_phase state = barrier_phase::collecting;
    commit_certificate certificate;
    digest certificate_digest{};
    uint8_t applied_mask = 0, committed_mask = 0;

    void check_owner(uint32_t authenticated_rank, const owner_receipt & receipt) const;
    [[noreturn]] void conflict();
};

} // namespace glm_rpc
