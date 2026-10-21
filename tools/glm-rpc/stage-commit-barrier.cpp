#include "stage-commit-barrier.h"

#include <stdexcept>

namespace glm_rpc {

commit_barrier::commit_barrier(const step_key & expected, const limits & bounds) : expected(expected), bounds(bounds) {
    validate_step(expected, bounds);
}

void commit_barrier::check_owner(uint32_t authenticated_rank, const owner_receipt & receipt) const {
    if (state == barrier_phase::invalid) { throw std::invalid_argument("commit barrier is permanently invalid"); }
    validate_receipt(receipt, bounds);
    if (authenticated_rank > 2 || receipt.rank != authenticated_rank || !(receipt.step == expected)) {
        throw std::invalid_argument("receipt differs from authenticated owner or pending coordinator key");
    }
}

void commit_barrier::invalidate() noexcept {
    state = barrier_phase::invalid;
    certificate = {}; certificate_digest = {};
    applied_mask = committed_mask = 0;
}

[[noreturn]] void commit_barrier::conflict() {
    invalidate();
    throw std::runtime_error("conflicting receipt invalidated the coordinator transaction");
}

void commit_barrier::record_applied(uint32_t authenticated_rank, const owner_receipt & receipt) {
    check_owner(authenticated_rank, receipt);
    const auto bit = uint8_t(1u << authenticated_rank);
    if (applied_mask & bit) {
        if (!(certificate.owners[authenticated_rank] == receipt)) { conflict(); }
        return;
    }
    certificate.owners[authenticated_rank] = receipt;
    applied_mask |= bit;
    if (applied_mask == 7) {
        try { certificate_digest = fingerprint_commit(certificate, bounds); }
        catch (...) { invalidate(); throw; }
        state = barrier_phase::decided;
    }
}

const commit_certificate & commit_barrier::decision() const {
    if (state != barrier_phase::decided && state != barrier_phase::complete) {
        throw std::invalid_argument("commit decision requires all three matching applied receipts");
    }
    return certificate;
}

void commit_barrier::record_committed(uint32_t authenticated_rank, const commit_acknowledgement & acknowledgement) {
    check_owner(authenticated_rank, acknowledgement.owner);
    if (state != barrier_phase::decided && state != barrier_phase::complete) {
        throw std::invalid_argument("commit acknowledgement arrived before a complete decision");
    }
    if (acknowledgement.certificate_digest != certificate_digest ||
            !(acknowledgement.owner == certificate.owners[authenticated_rank])) { conflict(); }
    committed_mask |= uint8_t(1u << authenticated_rank);
    if (committed_mask == 7) { state = barrier_phase::complete; }
}

const commit_certificate & commit_barrier::committed() const {
    if (state != barrier_phase::complete) { throw std::invalid_argument("global commit requires all three distinct commit acknowledgements"); }
    return certificate;
}

} // namespace glm_rpc
