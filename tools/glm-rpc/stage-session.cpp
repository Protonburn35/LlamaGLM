#include "stage-session.h"

#include <algorithm>
#include <stdexcept>

namespace glm_rpc {
namespace {

template<size_t N> bool nonzero(const std::array<uint8_t, N> & value) {
    return std::any_of(value.begin(), value.end(), [](uint8_t byte) { return byte != 0; });
}

bool same(const epoch_stamp & a, const epoch_stamp & b) {
    return a.identity == b.identity && a.session == b.session && a.epoch == b.epoch && a.generation == b.generation;
}

bool same(const step_key & a, const step_key & b) {
    return same(a.stamp, b.stamp) && a.transaction == b.transaction && a.chunk == b.chunk &&
        a.position == b.position && a.rows == b.rows && a.request_digest == b.request_digest;
}

bool same(const owner_receipt & a, const owner_receipt & b) {
    return same(a.step, b.step) && a.rank == b.rank && a.end_position == b.end_position &&
        a.input_digest == b.input_digest && a.output_digest == b.output_digest;
}

bool same(const commit_certificate & a, const commit_certificate & b) {
    for (size_t rank = 0; rank < 3; ++rank) { if (!same(a.owners[rank], b.owners[rank])) { return false; } }
    return true;
}

void check_stamp(const epoch_stamp & stamp) {
    if (!nonzero(stamp.identity) || !nonzero(stamp.session) || stamp.epoch == 0 || stamp.generation == 0 ||
            stamp.epoch == UINT64_MAX || stamp.generation == UINT64_MAX) {
        throw std::invalid_argument("invalid session epoch/generation/identity");
    }
}

void check_key(const step_key & step, const limits & bounds) {
    check_stamp(step.stamp);
    if (!nonzero(step.request_digest) || step.transaction == 0 || step.transaction == UINT64_MAX ||
            step.chunk == 0 || step.chunk == UINT64_MAX || step.rows == 0 || step.rows > bounds.rows ||
            step.position > bounds.context || step.rows > bounds.context - step.position) {
        throw std::invalid_argument("invalid transaction coordinates/digest or context capacity");
    }
}

} // namespace

bool operator==(const step_key & a, const step_key & b) { return same(a, b); }
bool operator==(const owner_receipt & a, const owner_receipt & b) { return same(a, b); }

void validate_step(const step_key & step, const limits & bounds) {
    bounds.validate();
    check_key(step, bounds);
}

void validate_receipt(const owner_receipt & receipt, const limits & bounds) {
    bounds.validate();
    check_key(receipt.step, bounds);
    if (receipt.rank > 2 || receipt.end_position != receipt.step.position + receipt.step.rows ||
            !nonzero(receipt.input_digest) || !nonzero(receipt.output_digest) ||
            (receipt.rank == 0 && receipt.input_digest != receipt.step.request_digest)) {
        throw std::invalid_argument("invalid owner applied-state receipt");
    }
}

void validate_commit(const commit_certificate & certificate, const limits & bounds) {
    bounds.validate();
    const auto & step = certificate.owners[0].step;
    check_key(step, bounds);
    for (uint32_t rank = 0; rank < 3; ++rank) {
        const auto & receipt = certificate.owners[rank];
        validate_receipt(receipt, bounds);
        if (receipt.rank != rank || !same(receipt.step, step)) {
            throw std::invalid_argument("commit requires complete matching A/B/C owner receipts");
        }
        const auto & expected = rank ? certificate.owners[rank-1].output_digest : step.request_digest;
        if (receipt.input_digest != expected) { throw std::invalid_argument("commit owner input/output digest chain differs"); }
    }
}

owner_transactions::owner_transactions(uint32_t rank, const digest & identity, const limits & bounds) :
    rank(rank), identity(identity), bounds(bounds) {
    bounds.validate();
    if (rank > 2 || !nonzero(identity)) { throw std::invalid_argument("invalid transaction owner or execution identity"); }
}

void owner_transactions::fresh_context(const epoch_stamp & stamp) {
    check_stamp(stamp);
    if ((state != owner_phase::empty && state != owner_phase::invalid) || stamp.identity != identity ||
            stamp.epoch <= current.epoch || stamp.generation <= current.generation) {
        throw std::invalid_argument("fresh context requires invalid state and strictly newer epoch/generation");
    }
    current = stamp;
    position = 0; next_transaction = next_chunk = 1;
    pending.reset(); last.reset();
    state = owner_phase::idle;
}

void owner_transactions::check_epoch(const epoch_stamp & stamp) const {
    if (state == owner_phase::empty || state == owner_phase::invalid || !same(stamp, current)) {
        throw std::invalid_argument("inactive or mismatched session epoch/generation/identity");
    }
}

void owner_transactions::check_step(const step_key & step) const {
    check_epoch(step.stamp);
    check_key(step, bounds);
}

void owner_transactions::invalidate() noexcept {
    state = owner_phase::invalid;
    pending.reset(); last.reset();
}

[[noreturn]] void owner_transactions::conflict() {
    invalidate();
    throw std::runtime_error("conflicting transaction retry invalidated this owner epoch");
}

step_disposition owner_transactions::prepare(const step_key & step, const digest & input_digest) {
    check_step(step);
    if (!nonzero(input_digest) || (rank == 0 && input_digest != step.request_digest)) {
        throw std::invalid_argument("invalid owner input digest");
    }
    if (last && step.transaction == last->owners[rank].step.transaction) {
        const auto & receipt = last->owners[rank];
        if (!same(step, receipt.step) || input_digest != receipt.input_digest) { conflict(); }
        return step_disposition::committed;
    }
    if (pending && step.transaction == pending->step.transaction) {
        if (!same(step, pending->step) || input_digest != pending->input_digest) { conflict(); }
        return state == owner_phase::applied ? step_disposition::applied : step_disposition::running;
    }
    if (state != owner_phase::idle || step.transaction != next_transaction || step.chunk != next_chunk || step.position != position) {
        throw std::invalid_argument("transaction busy, stale/expired, skipped or out of position");
    }
    pending = owner_receipt{step, rank, step.position + step.rows, input_digest, {}};
    state = owner_phase::running;
    return step_disposition::execute;
}

owner_receipt owner_transactions::applied(const step_key & step, const digest & output_digest) {
    check_step(step);
    if (!nonzero(output_digest)) { throw std::invalid_argument("invalid owner output digest"); }
    if (last && step.transaction == last->owners[rank].step.transaction) {
        const auto & receipt = last->owners[rank];
        if (!same(step, receipt.step) || output_digest != receipt.output_digest) { conflict(); }
        return receipt;
    }
    if (!pending || step.transaction != pending->step.transaction) { throw std::invalid_argument("transaction was not prepared"); }
    if (!same(step, pending->step)) { conflict(); }
    if (state == owner_phase::applied) {
        if (output_digest != pending->output_digest) { conflict(); }
        return *pending;
    }
    if (state != owner_phase::running) { throw std::invalid_argument("transaction is not running"); }
    pending->output_digest = output_digest;
    state = owner_phase::applied;
    return *pending;
}

owner_receipt owner_transactions::commit(const commit_certificate & certificate) {
    const auto & step = certificate.owners[rank].step;
    check_step(step);
    if (last && step.transaction == last->owners[rank].step.transaction) {
        if (!same(certificate, *last)) { conflict(); }
        return last->owners[rank];
    }
    if (!pending || state != owner_phase::applied || step.transaction != pending->step.transaction) {
        throw std::invalid_argument("commit requires a completely applied local transaction");
    }
    try { validate_commit(certificate, bounds); }
    catch (...) { invalidate(); throw; }
    if (!same(certificate.owners[rank], *pending)) { conflict(); }
    last = certificate;
    position = pending->end_position;
    ++next_transaction; ++next_chunk;
    pending.reset();
    state = owner_phase::idle;
    return last->owners[rank];
}

} // namespace glm_rpc
