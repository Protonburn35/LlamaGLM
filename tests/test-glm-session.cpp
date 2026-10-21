#include "stage-session.h"
#include "stage-session-codec.h"
#include "stage-commit-barrier.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <limits>
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

static commit_certificate codec_fixture() {
    step_key key;
    for (size_t i = 0; i < 32; ++i) { key.stamp.identity[i] = uint8_t(i); key.request_digest[i] = uint8_t(0x20 + i); }
    for (size_t i = 0; i < 16; ++i) { key.stamp.session[i] = uint8_t(0xa0 + i); }
    key.stamp.epoch = 0x0102030405060708ull; key.stamp.generation = 0x1112131415161718ull;
    key.transaction = 0x2122232425262728ull; key.chunk = 0x3132333435363738ull;
    key.position = 17; key.rows = 4;
    commit_certificate result;
    for (uint32_t rank = 0; rank < 3; ++rank) {
        auto & receipt = result.owners[rank];
        receipt.step = key; receipt.rank = rank; receipt.end_position = 21;
        receipt.input_digest = rank ? result.owners[rank-1].output_digest : key.request_digest;
        for (size_t i = 0; i < 32; ++i) { receipt.output_digest[i] = uint8_t(0x40 + rank * 32 + i); }
    }
    return result;
}

static std::string hex(const digest & value) {
    const char * digits = "0123456789abcdef";
    std::string result;
    for (const auto byte : value) { result.push_back(digits[byte >> 4]); result.push_back(digits[byte & 15]); }
    return result;
}

static void codec_golden() {
    const auto fixture = codec_fixture();
    const auto bounds = small_bounds();
    const auto receipt = encode_applied_receipt(fixture.owners[1], bounds);
    const auto certificate = encode_commit_certificate(fixture, bounds);
    // Independent Python struct.pack('<...') + hashlib vectors, including integers above 32 bits.
    require(hex(sha256({{receipt.data(), receipt.size()}})) == "e1339b7559be2f1760dd72ceb9f4d0a5f06e204ac053c8a6310fe04222b96d75" &&
            hex(sha256({{certificate.data(), certificate.size()}})) == "baa3e6661fc8dd4ea41d4c558c72d1620d841aa4b1ee217e2d77475676f05544",
            "session codec differs from independent little-endian golden vectors");
    const auto decoded = decode_applied_receipt({receipt.data(), receipt.size()}, 1, fixture.owners[1].step, bounds);
    require(decoded.step == fixture.owners[1].step && decoded.rank == 1 && decoded.end_position == 21 &&
            decoded.input_digest == fixture.owners[1].input_digest && decoded.output_digest == fixture.owners[1].output_digest,
            "owner receipt round trip changed semantic fields");
    const auto committed = decode_commit_certificate({certificate.data(), certificate.size()}, fixture.owners[0].step, bounds);
    require(encode_commit_certificate(committed, bounds) == certificate, "complete commit certificate round trip differs");
}

static void codec_refusals() {
    const auto fixture = codec_fixture();
    const auto bounds = small_bounds();
    const auto & expected = fixture.owners[0].step;
    const auto receipt = encode_applied_receipt(fixture.owners[1], bounds);
    const auto certificate = encode_commit_certificate(fixture, bounds);
    refused([&] { decode_applied_receipt({nullptr, receipt.size()}, 1, expected, bounds); });
    refused([&] { decode_commit_certificate({nullptr, certificate.size()}, expected, bounds); });
    for (size_t size = 0; size < receipt.size(); ++size) {
        refused([&] { decode_applied_receipt({receipt.data(), size}, 1, expected, bounds); });
    }
    for (size_t size = 0; size < certificate.size(); ++size) {
        refused([&] { decode_commit_certificate({certificate.data(), size}, expected, bounds); });
    }
    refused([&] { decode_applied_receipt({receipt.data(), receipt.size() + 1}, 1, expected, bounds); });
    refused([&] { decode_commit_certificate({certificate.data(), certificate.size() + 1}, expected, bounds); });
    refused([&] { decode_applied_receipt({receipt.data(), receipt.size()}, 0, expected, bounds); });
    refused([&] { decode_applied_receipt({receipt.data(), receipt.size()}, 3, expected, bounds); });
    for (const size_t offset : {size_t(0), size_t(8), size_t(10), size_t(12), size_t(16), size_t(48),
            size_t(64), size_t(72), size_t(80), size_t(88), size_t(96), size_t(104), size_t(108), size_t(112),
            size_t(144), size_t(148), size_t(152)}) {
        auto changed = receipt; changed[offset] ^= 0x80;
        refused([&] { decode_applied_receipt({changed.data(), changed.size()}, 1, expected, bounds); });
    }
    for (const size_t offset : {size_t(0), size_t(8), size_t(10), size_t(12), size_t(16), size_t(48),
            size_t(64), size_t(72), size_t(80), size_t(88), size_t(96), size_t(104), size_t(108), size_t(112),
            size_t(144), size_t(148), size_t(152), size_t(160), size_t(192), size_t(224), size_t(228),
            size_t(232), size_t(240), size_t(272), size_t(304), size_t(308), size_t(312), size_t(320)}) {
        auto changed = certificate; changed[offset] ^= 0x80;
        refused([&] { decode_commit_certificate({changed.data(), changed.size()}, expected, bounds); });
    }
    auto wrong_key = expected; ++wrong_key.stamp.epoch;
    refused([&] { decode_commit_certificate({certificate.data(), certificate.size()}, wrong_key, bounds); });
    refused([&] { encode_applied_receipt({}, bounds); });
    auto missing = fixture; missing.owners[1] = {};
    refused([&] { encode_commit_certificate(missing, bounds); });
    uint32_t random = 53;
    for (size_t iteration = 0; iteration < 2000; ++iteration) {
        auto changed = certificate;
        for (size_t byte = 0; byte <= iteration % 8; ++byte) {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            changed[random % changed.size()] ^= uint8_t((random >> 16) | 1);
        }
        try {
            const auto parsed = decode_commit_certificate({changed.data(), changed.size()}, expected, bounds);
            require(encode_commit_certificate(parsed, bounds) == changed, "accepted session payload has ignored/noncanonical bytes");
        } catch (const std::invalid_argument &) {}
    }
}

static void authenticated_certificate() {
    const auto certificate = codec_fixture();
    const auto & expected = certificate.owners[0].step;
    const auto bounds = small_bounds();
    const auto payload = encode_commit_certificate(certificate, bounds);
    auto psk = random_nonce();
    binding peer{random_nonce(), random_nonce(), expected.stamp.identity, 0, 1, channel::control};
    auto key = derive_link_key(psk, peer); erase_secret(psk);
    sender tx(key, peer, bounds);
    receiver good(key, peer, bounds), damaged(key, peer, bounds);
    header metadata;
    metadata.type = kind::commit; metadata.session = expected.stamp.session;
    metadata.epoch = expected.stamp.epoch; metadata.generation = expected.stamp.generation;
    metadata.transaction = expected.transaction; metadata.chunk = expected.chunk;
    metadata.position = expected.position;
    metadata.payload_bytes = uint32_t(payload.size()); metadata.total_bytes = payload.size();
    auto packet = tx.encode(metadata, {payload.data(), payload.size()});
    wire_header raw{}; std::copy_n(packet.begin(), raw.size(), raw.begin());
    const auto inspected = good.authenticate(raw, {packet.data() + raw.size(), payload.size()});
    require(inspected.type == kind::commit && encode_commit_certificate(decode_commit_decision(inspected,
            {packet.data() + raw.size(), payload.size()}, expected, bounds), bounds) == payload,
            "authenticated control frame changed its commit certificate");
    for (int field = 0; field < 19; ++field) {
        auto changed = inspected;
        if (field == 0) { changed.type = kind::state_ack; }
        if (field == 1) { changed.sender = 3; }
        if (field == 2) { changed.receiver = changed.sender; }
        if (field == 3) { changed.session[0] ^= 1; }
        if (field == 4) { ++changed.epoch; }
        if (field == 5) { ++changed.generation; }
        if (field == 6) { ++changed.transaction; }
        if (field == 7) { ++changed.chunk; }
        if (field == 8) { ++changed.position; }
        if (field == 9) { changed.rows = 1; }
        if (field == 10) { changed.hidden = 128; }
        if (field == 11) { changed.streams = 4; }
        if (field == 12) { changed.element = dtype::f32; }
        if (field == 13) { changed.fragment = 1; }
        if (field == 14) { changed.fragments = 2; }
        if (field == 15) { ++changed.payload_bytes; }
        if (field == 16) { ++changed.total_bytes; }
        if (field == 17) { changed.sequence = 0; }
        if (field == 18) { changed.sender = 2; }
        refused([&] { decode_commit_decision(changed, {payload.data(), payload.size()}, expected, bounds); });
    }
    const auto applied = encode_applied_receipt(certificate.owners[1], bounds);
    auto acknowledgement = inspected; acknowledgement.type = kind::state_ack; acknowledgement.sender = 1; acknowledgement.receiver = 0;
    acknowledgement.payload_bytes = uint32_t(applied.size()); acknowledgement.total_bytes = applied.size();
    require(decode_applied_ack(acknowledgement, {applied.data(), applied.size()}, expected, bounds).rank == 1,
            "applied acknowledgement lost its authenticated owner");
    auto wrong_receiver = acknowledgement; wrong_receiver.receiver = 2;
    refused([&] { decode_applied_ack(wrong_receiver, {applied.data(), applied.size()}, expected, bounds); });
    const auto terminal = encode_applied_receipt(certificate.owners[2], bounds);
    auto from_c = acknowledgement; from_c.sender = 2;
    require(decode_applied_ack(from_c, {terminal.data(), terminal.size()}, expected, bounds).rank == 2,
            "terminal applied acknowledgement did not reach coordinator A");
    acknowledgement.type = kind::commit;
    refused([&] { decode_applied_ack(acknowledgement, {applied.data(), applied.size()}, expected, bounds); });
    // Changing only C's output remains structurally possible; MAC verification must precede decoding.
    packet.back() ^= 1;
    bool decoded = false;
    refused([&] {
        damaged.authenticate(raw, {packet.data() + raw.size(), payload.size()});
        decode_commit_decision(inspected, {packet.data() + raw.size(), payload.size()}, expected, bounds);
        decoded = true;
    });
    require(!decoded, "unauthenticated certificate reached its decoder");
    refused([&] { damaged.authenticate(raw, {payload.data(), payload.size()}); });
    erase_secret(key);
}

static void global_barrier_permutations() {
    std::array<uint32_t, 3> apply_order{0, 1, 2};
    do {
        std::array<uint32_t, 3> commit_order{0, 1, 2};
        do {
            owners group;
            const auto certificate = group.apply(step_for(group.stamp));
            commit_barrier barrier(certificate.owners[0].step, group.bounds);
            refused([&] { barrier.decision(); });
            refused([&] { barrier.committed(); });
            for (size_t i = 0; i < 3; ++i) {
                const auto rank = apply_order[i];
                barrier.record_applied(rank, certificate.owners[rank]);
                barrier.record_applied(rank, certificate.owners[rank]);
                refused([&] { barrier.committed(); });
                require(barrier.phase() == (i < 2 ? barrier_phase::collecting : barrier_phase::decided),
                        "duplicate applied receipt bypassed the three-owner decision barrier");
                if (i < 2) { refused([&] { barrier.decision(); }); }
            }
            require(fingerprint_commit(barrier.decision(), group.bounds) == fingerprint_commit(certificate, group.bounds),
                    "receipt collection changed the complete decision");
            for (size_t i = 0; i < 3; ++i) {
                const auto rank = commit_order[i];
                const auto acknowledgement = make_commit_ack(group.values[rank]->commit(barrier.decision()), certificate, group.bounds);
                barrier.record_committed(rank, acknowledgement);
                barrier.record_committed(rank, acknowledgement);
                barrier.record_applied(rank, certificate.owners[rank]);
                require(barrier.phase() == (i < 2 ? barrier_phase::decided : barrier_phase::complete),
                        "duplicate commit ACK counted as another owner");
                if (i < 2) { refused([&] { barrier.committed(); }); }
            }
            require(barrier.committed().owners[2] == certificate.owners[2], "global barrier released a different C result");
            barrier.invalidate();
            refused([&] { barrier.committed(); });
            refused([&] { barrier.record_applied(0, certificate.owners[0]); });
            require(barrier.phase() == barrier_phase::invalid, "invalidated barrier revived");
        } while (std::next_permutation(commit_order.begin(), commit_order.end()));
    } while (std::next_permutation(apply_order.begin(), apply_order.end()));
}

static void global_barrier_refusals() {
    owners group;
    const auto certificate = group.apply(step_for(group.stamp));
    const auto & step = certificate.owners[0].step;
    commit_barrier barrier(step, group.bounds);
    const auto acknowledgement = make_commit_ack(certificate.owners[0], certificate, group.bounds);
    refused([&] { barrier.record_committed(0, acknowledgement); });
    refused([&] { barrier.record_applied(1, certificate.owners[0]); });
    refused([&] { barrier.record_applied(3, certificate.owners[0]); });
    auto stale = certificate.owners[0]; ++stale.step.stamp.epoch;
    refused([&] { barrier.record_applied(0, stale); });
    require(barrier.phase() == barrier_phase::collecting, "rejected uncorrelated fact mutated the barrier");
    barrier.record_applied(0, certificate.owners[0]);
    auto changed = certificate.owners[0]; changed.output_digest[0] ^= 1;
    refused([&] { barrier.record_applied(0, changed); });
    require(barrier.phase() == barrier_phase::invalid, "conflicting applied result did not invalidate barrier");
    for (int field = 0; field < 3; ++field) {
        commit_barrier checked(step, group.bounds);
        for (uint32_t rank = 0; rank < 3; ++rank) { checked.record_applied(rank, certificate.owners[rank]); }
        auto wrong = acknowledgement;
        if (field == 0) { wrong.certificate_digest[0] ^= 1; }
        if (field == 1) { wrong.owner.output_digest[0] ^= 1; }
        if (field == 2) { wrong.certificate_digest = {}; }
        refused([&] { checked.record_committed(0, wrong); });
        require(checked.phase() == barrier_phase::invalid, "conflicting committed ACK retained a usable barrier");
        refused([&] { checked.committed(); });
    }
    commit_barrier broken(step, group.bounds);
    auto disconnected = certificate.owners[1]; disconnected.input_digest[0] ^= 1;
    broken.record_applied(1, disconnected); broken.record_applied(0, certificate.owners[0]);
    refused([&] { broken.record_applied(2, certificate.owners[2]); });
    require(broken.phase() == barrier_phase::invalid, "broken digest chain produced a decision");
    // B never ACKs. Repeating A/C cannot publish; lost ACK invalidates all owners, even
    // ones that committed locally. A fresh barrier/context epoch must replay from zero.
    commit_barrier uncertain(step, group.bounds);
    for (uint32_t rank = 0; rank < 3; ++rank) { uncertain.record_applied(rank, certificate.owners[rank]); }
    for (uint32_t rank : {0u, 2u}) {
        const auto ack = make_commit_ack(group.values[rank]->commit(uncertain.decision()), certificate, group.bounds);
        uncertain.record_committed(rank, ack); uncertain.record_committed(rank, ack);
    }
    refused([&] { uncertain.committed(); });
    uncertain.invalidate();
    const auto fresh = stamp_for(2, 2);
    for (auto & owner : group.values) { owner->invalidate(); owner->fresh_context(fresh); }
    const auto replayed = group.apply(step_for(fresh));
    commit_barrier replay(replayed.owners[0].step, group.bounds);
    refused([&] { replay.record_applied(0, certificate.owners[0]); });
    for (uint32_t rank = 0; rank < 3; ++rank) { replay.record_applied(rank, replayed.owners[rank]); }
    for (uint32_t rank = 0; rank < 3; ++rank) {
        replay.record_committed(rank, make_commit_ack(group.values[rank]->commit(replay.decision()), replayed, group.bounds));
    }
    require(replay.committed().owners[2].end_position == 17, "fresh epoch did not complete replay");
}

static void committed_ack_codec() {
    const auto certificate = codec_fixture();
    const auto bounds = small_bounds();
    const auto & expected = certificate.owners[0].step;
    const auto acknowledgement = make_commit_ack(certificate.owners[1], certificate, bounds);
    const auto payload = encode_commit_ack(acknowledgement, bounds);
    require(hex(sha256({{payload.data(), payload.size()}})) == "5280d8a96e117e039d307e7ee6d2248bbc5a1e502c9cc84e999825098fce0af1",
            "commit ACK differs from independent Python little-endian golden bytes");
    require(decode_commit_ack({payload.data(), payload.size()}, 1, expected, acknowledgement.certificate_digest, bounds).owner == acknowledgement.owner,
            "commit ACK round trip lost the local committed receipt");
    for (size_t length = 0; length < payload.size(); ++length) {
        refused([&] { decode_commit_ack({payload.data(), length}, 1, expected, acknowledgement.certificate_digest, bounds); });
    }
    refused([&] { decode_commit_ack({nullptr, payload.size()}, 1, expected, acknowledgement.certificate_digest, bounds); });
    for (size_t offset : {size_t(0), size_t(8), size_t(10), size_t(12), size_t(108), size_t(148), size_t(224), size_t(255)}) {
        auto wrong = payload; wrong[offset] ^= 1;
        refused([&] { decode_commit_ack({wrong.data(), wrong.size()}, 1, expected, acknowledgement.certificate_digest, bounds); });
    }
    const auto applied = encode_applied_receipt(certificate.owners[1], bounds);
    refused([&] { decode_commit_ack({applied.data(), applied.size()}, 1, expected, acknowledgement.certificate_digest, bounds); });
    refused([&] { decode_applied_receipt({payload.data(), payload.size()}, 1, expected, bounds); });
    auto key = random_nonce();
    binding peer{random_nonce(), random_nonce(), expected.stamp.identity, 1, 0, channel::control};
    auto derived = derive_link_key(key, peer); erase_secret(key);
    sender tx(derived, peer, bounds); receiver rx(derived, peer, bounds);
    header metadata; metadata.type = kind::commit_ack; metadata.sender = 1; metadata.receiver = 0;
    metadata.session = expected.stamp.session; metadata.epoch = expected.stamp.epoch; metadata.generation = expected.stamp.generation;
    metadata.transaction = expected.transaction; metadata.chunk = expected.chunk; metadata.position = expected.position;
    metadata.payload_bytes = uint32_t(payload.size()); metadata.total_bytes = payload.size();
    const auto packet = tx.encode(metadata, {payload.data(), payload.size()});
    wire_header raw{}; std::copy_n(packet.begin(), raw.size(), raw.begin());
    auto verified = rx.authenticate(raw, {packet.data() + raw.size(), payload.size()});
    require(decode_committed_ack(verified, {packet.data() + raw.size(), payload.size()}, expected,
            acknowledgement.certificate_digest, bounds).owner.rank == 1, "authenticated commit ACK lost its rank");
    verified.type = kind::state_ack;
    refused([&] { decode_committed_ack(verified, {payload.data(), payload.size()}, expected, acknowledgement.certificate_digest, bounds); });
    verified.type = kind::commit_ack; verified.receiver = 2;
    refused([&] { decode_committed_ack(verified, {payload.data(), payload.size()}, expected, acknowledgement.certificate_digest, bounds); });
    const auto completion = encode_commit_completion(certificate, bounds);
    require(hex(sha256({{completion.data(), completion.size()}})) == "c0d58358fd4e148fb28ac857d503f2f7da26a7948e378c1415bcc8d5337c8b26",
            "global completion differs from independent Python golden bytes");
    auto completed = verified; completed.sender = 0; completed.receiver = 1; completed.type = kind::commit_complete;
    completed.payload_bytes = uint32_t(completion.size()); completed.total_bytes = completion.size();
    decode_commit_completion(completed, {completion.data(), completion.size()}, expected, acknowledgement.certificate_digest, bounds);
    for (size_t length = 0; length < completion.size(); ++length) {
        refused([&] { decode_commit_completion(completed, {completion.data(), length}, expected, acknowledgement.certificate_digest, bounds); });
    }
    for (size_t offset : {size_t(0), size_t(8), size_t(10), size_t(12), size_t(16), size_t(108), size_t(144), size_t(175)}) {
        auto wrong = completion; wrong[offset] ^= 1;
        refused([&] { decode_commit_completion(completed, {wrong.data(), wrong.size()}, expected, acknowledgement.certificate_digest, bounds); });
    }
    completed.sender = 2;
    refused([&] { decode_commit_completion(completed, {completion.data(), completion.size()}, expected, acknowledgement.certificate_digest, bounds); });
    completed.sender = 0; completed.type = kind::commit;
    refused([&] { decode_commit_completion(completed, {completion.data(), completion.size()}, expected, acknowledgement.certificate_digest, bounds); });
    erase_secret(derived);
}

static void terminal_codec() {
    const auto certificate = codec_fixture();
    const auto bounds = small_bounds();
    terminal_outcome result; result.step = certificate.owners[0].step;
    result.certificate_digest = fingerprint_commit(certificate,bounds); result.vocabulary = 128;
    result.selected_token = 5; result.selected_logit = 1.25f; result.count = 2; result.row = 3;
    result.top[0] = {5,1.25f}; result.top[1] = {7,0.5f};
    const auto payload = encode_terminal_outcome(result,bounds);
    require(hex(sha256({{payload.data(),payload.size()}})) == "abd5a28d247dbd2840987d290c14db2a5e1fbc8292a63fd77e30fd31f7488b47",
            "terminal outcome differs from independent Python golden bytes");
    header metadata; metadata.type = kind::terminal_result; metadata.sender = 2; metadata.receiver = 0;
    metadata.session = result.step.stamp.session; metadata.epoch = result.step.stamp.epoch; metadata.generation = result.step.stamp.generation;
    metadata.transaction = result.step.transaction; metadata.chunk = result.step.chunk; metadata.position = result.step.position;
    metadata.payload_bytes = uint32_t(payload.size()); metadata.total_bytes = payload.size(); metadata.sequence = 1;
    const auto decode = [&](const header & envelope, bytes_view input) {
        return decode_terminal_outcome(envelope,input,result.step,result.certificate_digest,128,2,bounds);
    };
    require(encode_terminal_outcome(decode(metadata,{payload.data(),payload.size()}),bounds) == payload, "terminal round trip differs");
    for (size_t length = 0; length < payload.size(); ++length) {
        auto short_header = metadata; short_header.payload_bytes = uint32_t(length); short_header.total_bytes = length;
        refused([&] { decode(short_header,{payload.data(),length}); });
    }
    refused([&] { decode(metadata,{nullptr,payload.size()}); });
    for (size_t offset : {size_t(0),size_t(8),size_t(10),size_t(12),size_t(16),size_t(144),size_t(176),size_t(180),
            size_t(184),size_t(188),size_t(192),size_t(196),size_t(200),size_t(204),size_t(208),size_t(212),size_t(216),size_t(224),size_t(271)}) {
        auto wrong = payload; wrong[offset] ^= 0x80;
        refused([&] { decode(metadata,{wrong.data(),wrong.size()}); });
    }
    auto wrong_role = metadata; wrong_role.sender = 1;
    refused([&] { decode(wrong_role,{payload.data(),payload.size()}); });
    auto wrong_phase = metadata; wrong_phase.type = kind::state_ack;
    refused([&] { decode(wrong_phase,{payload.data(),payload.size()}); });
    refused([&] { decode_terminal_outcome(metadata,{payload.data(),payload.size()},result.step,result.certificate_digest,127,2,bounds); });
    refused([&] { decode_terminal_outcome(metadata,{payload.data(),payload.size()},result.step,result.certificate_digest,128,8,bounds); });
    for (int field = 0; field < 6; ++field) {
        auto wrong = result;
        if (field == 0) { wrong.mode = 2; }
        if (field == 1) { wrong.top[1].token = wrong.top[0].token; }
        if (field == 2) { wrong.top[1].logit = 2; }
        if (field == 3) { wrong.top[1].logit = std::numeric_limits<float>::infinity(); }
        if (field == 4) { wrong.top[1].logit = std::numeric_limits<float>::quiet_NaN(); }
        if (field == 5) { wrong.top[1] = {4,wrong.top[0].logit}; }
        refused([&] { encode_terminal_outcome(wrong,bounds); });
    }
}

static void begin_codec() {
    const auto fixture = codec_fixture();
    const auto bounds = small_bounds();
    const auto & step = fixture.owners[0].step;
    const begin_coordinates expected{step.stamp,step.transaction,step.chunk,step.position,step.rows};
    const auto payload = encode_step_begin(step,bounds);
    require(hex(sha256({{payload.data(),payload.size()}})) == "53747be8093834c18af123596fc5014e835ec1ef398992f02aeaf774e21bcf90",
            "BEGIN differs from independent Python little-endian golden bytes");
    header metadata; metadata.type = kind::begin; metadata.sender = 0; metadata.receiver = 1;
    metadata.session = step.stamp.session; metadata.epoch = step.stamp.epoch; metadata.generation = step.stamp.generation;
    metadata.transaction = step.transaction; metadata.chunk = step.chunk; metadata.position = step.position;
    metadata.payload_bytes = uint32_t(payload.size()); metadata.total_bytes = payload.size(); metadata.sequence = 1;
    require(decode_step_begin(metadata,{payload.data(),payload.size()},expected,bounds) == step, "BEGIN lost its full request key");
    for (size_t length = 0; length < payload.size(); ++length) {
        auto short_header = metadata; short_header.payload_bytes = uint32_t(length); short_header.total_bytes = length;
        refused([&] { decode_step_begin(short_header,{payload.data(),length},expected,bounds); });
    }
    refused([&] { decode_step_begin(metadata,{nullptr,payload.size()},expected,bounds); });
    for (size_t offset : {size_t(0),size_t(8),size_t(10),size_t(12),size_t(16),size_t(48),size_t(64),size_t(72),
            size_t(80),size_t(88),size_t(96),size_t(104),size_t(108)}) {
        auto wrong = payload; wrong[offset] ^= 0x80;
        refused([&] { decode_step_begin(metadata,{wrong.data(),wrong.size()},expected,bounds); });
    }
    auto wrong_role = metadata; wrong_role.sender = 2;
    refused([&] { decode_step_begin(wrong_role,{payload.data(),payload.size()},expected,bounds); });
    auto wrong_phase = metadata; wrong_phase.type = kind::commit;
    refused([&] { decode_step_begin(wrong_phase,{payload.data(),payload.size()},expected,bounds); });
    auto changed_request = step; changed_request.request_digest[0] ^= 1;
    const auto different = encode_step_begin(changed_request,bounds);
    require(decode_step_begin(metadata,{different.data(),different.size()},expected,bounds) == changed_request,
            "BEGIN incorrectly required workers to know A's new token request");
    owners committed;
    const auto original = committed.apply(step_for(committed.stamp)); committed.commit(original);
    auto conflict = original.owners[1].step; conflict.request_digest[0] ^= 1;
    refused([&] { committed.values[1]->prepare(conflict,original.owners[1].input_digest); });
    require(committed.values[1]->phase() == owner_phase::invalid, "changed duplicate BEGIN bypassed semantic request dedup");
}

int main() {
    try {
        happy_and_expired(); malformed_requests(); conflicts(); certificate_barrier(); uncertain_reset(); reset_and_capacity();
        codec_golden(); codec_refusals(); authenticated_certificate();
        global_barrier_permutations(); global_barrier_refusals(); committed_ack_codec();
        terminal_codec();
        begin_codec();
        std::puts("GLM session control: fourteen suites passed (owner lifecycle, codecs/authentication, BEGIN agreement, commit ACKs/barrier, reset/replay, compact greedy outcomes); production dispatch/publication NOT_IMPLEMENTED");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM session ledger test failed: %s\n", error.what()); return 1;
    }
}
