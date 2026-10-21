#include "stage-session.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace glm_rpc;

static void require(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }
template<class F> static void refused(F function) {
    try { function(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid session transition was accepted");
}

static digest fingerprint(const std::string & text) {
    return sha256({{reinterpret_cast<const uint8_t *>(text.data()), text.size()}});
}

static limits small_bounds() {
    limits result; result.hidden = 128; result.rows = 64; result.context = 256; result.frame_bytes = 64u << 10;
    return result;
}

static epoch_stamp stamp_for(uint64_t epoch = 1, uint64_t generation = 1) {
    epoch_stamp result{fingerprint("session fixture execution identity"), {}, epoch, generation};
    result.session[0] = 0x53;
    return result;
}

static step_key step_for(const epoch_stamp & stamp, uint64_t id = 1, uint64_t position = 0, uint32_t rows = 17) {
    return {stamp, id, id, position, rows, fingerprint("original request " + std::to_string(id))};
}

struct owners {
    limits bounds = small_bounds();
    epoch_stamp stamp = stamp_for();
    std::array<std::unique_ptr<owner_transactions>, 3> values;
    owners() {
        for (uint32_t rank = 0; rank < 3; ++rank) {
            values[rank] = std::make_unique<owner_transactions>(rank, stamp.identity, bounds);
            values[rank]->fresh_context(stamp);
        }
    }
    commit_certificate apply(const step_key & step) {
        commit_certificate certificate;
        auto input = step.request_digest;
        for (uint32_t rank = 0; rank < 3; ++rank) {
            auto & owner = *values[rank];
            require(owner.prepare(step, input) == step_disposition::execute && owner.phase() == owner_phase::running,
                    "fresh transaction did not acquire one execution slot");
            require(owner.prepare(step, input) == step_disposition::running && owner.committed_position() == step.position,
                    "running retry advanced committed state");
            const auto output = fingerprint("owner " + std::to_string(rank) + " transaction " + std::to_string(step.transaction));
            certificate.owners[rank] = owner.applied(step, output);
            require(owner.prepare(step, input) == step_disposition::applied &&
                    owner.applied(step, output).output_digest == output && owner.committed_position() == step.position,
                    "applied retry advanced committed state or lost its result digest");
            input = output;
        }
        return certificate;
    }
    void commit(const commit_certificate & certificate) {
        validate_commit(certificate, bounds);
        for (uint32_t rank = 0; rank < 3; ++rank) {
            auto & owner = *values[rank];
            const auto receipt = owner.commit(certificate);
            require(owner.phase() == owner_phase::idle && owner.committed_position() == receipt.end_position &&
                    receipt.rank == rank && receipt.input_digest == certificate.owners[rank].input_digest &&
                    receipt.output_digest == certificate.owners[rank].output_digest,
                    "committed receipt differs from complete certificate");
            require(owner.commit(certificate).end_position == receipt.end_position &&
                    owner.prepare(receipt.step, receipt.input_digest) == step_disposition::committed &&
                    owner.committed_position() == receipt.end_position, "committed retry re-executed or advanced state");
        }
    }
};

static void happy_and_expired() {
    owners group;
    const auto first = group.apply(step_for(group.stamp));
    group.commit(first);
    const auto second = group.apply(step_for(group.stamp, 2, 17, 31));
    for (uint32_t rank = 0; rank < 3; ++rank) {
        auto & owner = *group.values[rank];
        require(owner.prepare(first.owners[rank].step, first.owners[rank].input_digest) == step_disposition::committed &&
                owner.commit(first).end_position == 17 && owner.phase() == owner_phase::applied && owner.committed_position() == 17,
                "previous committed retry displaced a newer pending transaction");
    }
    group.commit(second);
    for (uint32_t rank = 0; rank < 3; ++rank) {
        auto & owner = *group.values[rank];
        refused([&] { owner.prepare(first.owners[rank].step, first.owners[rank].input_digest); });
        refused([&] { owner.commit(first); });
        require(owner.phase() == owner_phase::idle && owner.committed_position() == 48,
                "expired receipt retry mutated a healthy owner");
    }
}

static void malformed_requests() {
    for (int field = 0; field < 17; ++field) {
        owners group;
        auto step = step_for(group.stamp);
        if (field == 0) { step.stamp.identity[0] ^= 1; }
        if (field == 1) { step.stamp.session[0] ^= 1; }
        if (field == 2) { ++step.stamp.epoch; }
        if (field == 3) { ++step.stamp.generation; }
        if (field == 4) { step.transaction = 0; }
        if (field == 5) { step.transaction = UINT64_MAX; }
        if (field == 6) { step.transaction = 2; }
        if (field == 7) { step.chunk = 0; }
        if (field == 8) { step.chunk = UINT64_MAX; }
        if (field == 9) { step.chunk = 2; }
        if (field == 10) { step.position = 1; }
        if (field == 11) { step.position = UINT64_MAX; }
        if (field == 12) { step.rows = 0; }
        if (field == 13) { step.rows = 65; }
        if (field == 14) { step.rows = UINT32_MAX; }
        if (field == 15) { step.request_digest = {}; }
        const auto input = field == 16 ? digest{} : step.request_digest;
        auto & owner = *group.values[0];
        refused([&] { owner.prepare(step, input); });
        require(owner.phase() == owner_phase::idle && owner.committed_position() == 0,
                "rejected request acquired mutable state");
    }
    owners group;
    const auto step = step_for(group.stamp);
    refused([&] { group.values[0]->applied(step, fingerprint("not executed")); });
    refused([&] { group.values[0]->prepare(step, fingerprint("different input")); });
}

static void conflicts() {
    for (int field = 0; field < 5; ++field) {
        owners group;
        auto key = step_for(group.stamp);
        auto & owner = *group.values[1];
        const auto input = fingerprint("A output");
        owner.prepare(key, input);
        if (field == 0) { ++key.chunk; }
        if (field == 1) { ++key.position; }
        if (field == 2) { ++key.rows; }
        if (field == 3) { key.request_digest[0] ^= 1; }
        if (field < 4) { refused([&] { owner.prepare(key, input); }); }
        else { refused([&] { owner.prepare(key, fingerprint("different A output")); }); }
        require(owner.phase() == owner_phase::invalid && owner.committed_position() == 0,
                "conflicting pending retry did not invalidate the epoch");
        refused([&] { owner.prepare(step_for(group.stamp), input); });
    }
    owners group;
    const auto step = step_for(group.stamp);
    auto & owner = *group.values[0];
    owner.prepare(step, step.request_digest);
    owner.applied(step, fingerprint("A output"));
    refused([&] { owner.applied(step, fingerprint("changed A output")); });
    require(owner.phase() == owner_phase::invalid, "conflicting applied result retained a usable epoch");
    owners committed;
    auto certificate = committed.apply(step_for(committed.stamp));
    committed.commit(certificate);
    certificate.owners[2].output_digest[0] ^= 1;
    refused([&] { committed.values[0]->commit(certificate); });
    require(committed.values[0]->phase() == owner_phase::invalid, "committed retry accepted a different terminal result");
}

static void certificate_barrier() {
    for (int field = 0; field < 10; ++field) {
        owners group;
        auto certificate = group.apply(step_for(group.stamp));
        if (field == 0) { certificate.owners[1] = {}; }
        if (field == 1) { certificate.owners[1] = certificate.owners[0]; }
        if (field == 2) { certificate.owners[2].rank = 1; }
        if (field == 3) { ++certificate.owners[1].step.chunk; }
        if (field == 4) { ++certificate.owners[1].step.position; }
        if (field == 5) { --certificate.owners[2].end_position; }
        if (field == 6) { certificate.owners[0].input_digest[0] ^= 1; }
        if (field == 7) { certificate.owners[1].input_digest[0] ^= 1; }
        if (field == 8) { certificate.owners[2].input_digest[0] ^= 1; }
        if (field == 9) { certificate.owners[2].output_digest = {}; }
        refused([&] { validate_commit(certificate, group.bounds); });
        refused([&] { group.values[0]->commit(certificate); });
        require(group.values[0]->phase() == owner_phase::invalid && group.values[0]->committed_position() == 0,
                "malformed complete-commit decision advanced local history");
    }
    owners premature;
    const auto step = step_for(premature.stamp);
    premature.values[0]->prepare(step, step.request_digest);
    refused([&] { premature.values[0]->commit({}); });
    require(premature.values[0]->committed_position() == 0, "commit without applied state advanced history");
    owners forged;
    auto certificate = forged.apply(step_for(forged.stamp));
    certificate.owners[0].output_digest = fingerprint("forged A result");
    certificate.owners[1].input_digest = certificate.owners[0].output_digest;
    validate_commit(certificate, forged.bounds);
    refused([&] { forged.values[0]->commit(certificate); });
    require(forged.values[0]->phase() == owner_phase::invalid && forged.values[0]->committed_position() == 0,
            "structurally valid certificate replaced the actual local result");
}

static void uncertain_reset() {
    owners group;
    const auto old = group.apply(step_for(group.stamp));
    group.values[0]->commit(old);
    group.values[1]->commit(old);
    // A lost commit acknowledgement cannot establish a global result. Reset every owner.
    for (uint32_t rank = 0; rank < 3; ++rank) {
        auto & owner = *group.values[rank];
        owner.invalidate(); owner.invalidate();
        refused([&] { owner.prepare(old.owners[rank].step, old.owners[rank].input_digest); });
        refused([&] { owner.commit(old); });
        auto fresh = stamp_for(2, 2); fresh.session[1] = 1;
        owner.fresh_context(fresh);
        require(owner.committed_position() == 0 && owner.phase() == owner_phase::idle,
                "fresh epoch retained uncertain positions");
        refused([&] { owner.prepare(old.owners[rank].step, old.owners[rank].input_digest); });
    }
    group.stamp = stamp_for(2, 2); group.stamp.session[1] = 1;
    group.commit(group.apply(step_for(group.stamp)));
}

static void reset_and_capacity() {
    static_assert(!std::is_copy_constructible<owner_transactions>::value, "mutable owner ledgers must not be copied");
    const auto stamp = stamp_for();
    refused([&] { owner_transactions invalid(3, stamp.identity, small_bounds()); });
    refused([&] { owner_transactions invalid(0, {}, small_bounds()); });
    auto bounds = small_bounds(); bounds.context = 1;
    owner_transactions owner(0, stamp.identity, bounds);
    refused([&] { owner.prepare(step_for(stamp), fingerprint("input")); });
    owner.fresh_context(stamp);
    refused([&] { owner.fresh_context(stamp_for(2, 2)); });
    owner.invalidate();
    for (int field = 0; field < 7; ++field) {
        auto fresh = stamp_for(2, 2);
        if (field == 0) { fresh.identity[0] ^= 1; }
        if (field == 1) { fresh.session = {}; }
        if (field == 2) { fresh.epoch = 1; }
        if (field == 3) { fresh.generation = 1; }
        if (field == 4) { fresh.epoch = UINT64_MAX; }
        if (field == 5) { fresh.generation = UINT64_MAX; }
        if (field == 6) { fresh.epoch = 0; }
        refused([&] { owner.fresh_context(fresh); });
        require(owner.phase() == owner_phase::invalid, "invalid reset revived an owner");
    }
    const auto fresh = stamp_for(2, 2);
    owner.fresh_context(fresh);
    const auto step = step_for(fresh, 1, 0, 1);
    owner.prepare(step, step.request_digest);
    commit_certificate certificate;
    certificate.owners[0] = owner.applied(step, fingerprint("A"));
    certificate.owners[1] = {step, 1, 1, certificate.owners[0].output_digest, fingerprint("B")};
    certificate.owners[2] = {step, 2, 1, certificate.owners[1].output_digest, fingerprint("C")};
    owner.commit(certificate);
    refused([&] { owner.prepare(step_for(fresh, 2, 1, 1), fingerprint("original request 2")); });
    require(owner.phase() == owner_phase::idle && owner.committed_position() == 1, "context overflow changed committed history");
}

int main() {
    try {
        happy_and_expired(); malformed_requests(); conflicts(); certificate_barrier(); uncertain_reset(); reset_and_capacity();
        std::puts("GLM session ledger: six suites passed (retry states, bounds, conflicts, A/B/C certificates, uncertain reset, epochs/capacity); metadata component only, global commit/transport/model integration NOT_IMPLEMENTED");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM session ledger test failed: %s\n", error.what()); return 1;
    }
}
