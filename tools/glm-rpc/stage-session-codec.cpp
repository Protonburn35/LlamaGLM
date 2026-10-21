#include "stage-session-codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace glm_rpc {
namespace {

constexpr std::array<uint8_t, 8> magic{'G','L','M','T','X','N',0,0};
constexpr uint16_t format_version = 1, receipt_kind = 1, commit_kind = 2, commit_ack_kind = 3, completion_kind = 4, terminal_kind = 5;
static_assert(commit_wire_bytes <= max_control_bytes, "transaction certificate must fit one bounded control frame");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "terminal logits require IEEE binary32");

uint32_t float_bits(float value) { uint32_t bits; std::memcpy(&bits,&value,sizeof(bits)); return bits; }
float from_bits(uint32_t bits) { float value; std::memcpy(&value,&bits,sizeof(value)); return value; }

void validate_terminal(const terminal_outcome & value, const limits & bounds) {
    validate_step(value.step,bounds);
    if (value.certificate_digest == digest{} || !value.vocabulary || value.vocabulary > (1u<<20) ||
            value.selected_token >= value.vocabulary || !std::isfinite(value.selected_logit) || value.mode != 1 ||
            !value.count || value.count > value.top.size() || value.count > value.vocabulary || value.row != value.step.rows-1) {
        throw std::invalid_argument("invalid stateless greedy terminal outcome");
    }
    for (size_t i = 0; i < value.top.size(); ++i) {
        const auto & candidate = value.top[i];
        if (i >= value.count) {
            if (candidate.token || float_bits(candidate.logit)) { throw std::invalid_argument("nonzero unused terminal candidate"); }
            continue;
        }
        if (candidate.token >= value.vocabulary || !std::isfinite(candidate.logit)) { throw std::invalid_argument("invalid terminal candidate"); }
        for (size_t previous = 0; previous < i; ++previous) {
            if (candidate.token == value.top[previous].token) { throw std::invalid_argument("duplicate terminal candidate token"); }
        }
        if (i && (candidate.logit > value.top[i-1].logit ||
                (candidate.logit == value.top[i-1].logit && candidate.token < value.top[i-1].token))) {
            throw std::invalid_argument("terminal candidates are not in canonical greedy order");
        }
    }
    if (value.selected_token != value.top[0].token || float_bits(value.selected_logit) != float_bits(value.top[0].logit)) {
        throw std::invalid_argument("selected greedy outcome differs from the first candidate");
    }
}

template<size_t N> struct writer {
    std::array<uint8_t, N> output{};
    size_t offset = 0;
    void room(size_t count) const {
        if (offset > N || count > N - offset) { throw std::logic_error("session encoder capacity exceeded"); }
    }
    void integer(uint64_t value, size_t bytes) {
        if (bytes > 8) { throw std::logic_error("session integer width exceeded"); }
        room(bytes);
        for (size_t i = 0; i < bytes; ++i) { output[offset++] = uint8_t(value); value >>= 8; }
    }
    template<size_t M> void block(const std::array<uint8_t, M> & value) {
        room(M); std::copy(value.begin(), value.end(), output.begin() + offset); offset += M;
    }
    void prefix(uint16_t kind) { block(magic); integer(format_version, 2); integer(kind, 2); integer(0, 4); }
    void key(const step_key & value) {
        block(value.stamp.identity); block(value.stamp.session);
        integer(value.stamp.epoch, 8); integer(value.stamp.generation, 8);
        integer(value.transaction, 8); integer(value.chunk, 8); integer(value.position, 8);
        integer(value.rows, 4); integer(0, 4); block(value.request_digest);
    }
    void owner(const owner_receipt & value) {
        integer(value.rank, 4); integer(0, 4); integer(value.end_position, 8);
        block(value.input_digest); block(value.output_digest);
    }
    std::array<uint8_t, N> finish() {
        if (offset != N) { throw std::logic_error("session encoder length differs"); }
        return output;
    }
};

struct reader {
    bytes_view input;
    size_t offset = 0;
    reader(bytes_view payload, size_t bytes, uint16_t kind) : input(payload) {
        if (!input.data || input.size != bytes) { throw std::invalid_argument("session payload requires its exact fixed length"); }
        std::array<uint8_t, 8> domain{}; block(domain);
        if (domain != magic || integer(2) != format_version || integer(2) != kind) {
            throw std::invalid_argument("unknown session payload domain/version/kind");
        }
        reserved();
    }
    void room(size_t count) const {
        if (offset > input.size || count > input.size - offset) { throw std::invalid_argument("truncated session payload"); }
    }
    uint64_t integer(size_t bytes) {
        if (bytes > 8) { throw std::invalid_argument("session integer width exceeded"); }
        room(bytes);
        uint64_t result = 0;
        for (size_t i = 0; i < bytes; ++i) { result |= uint64_t(input.data[offset++]) << (8 * i); }
        return result;
    }
    template<size_t N> void block(std::array<uint8_t, N> & value) {
        room(N); std::copy_n(input.data + offset, N, value.begin()); offset += N;
    }
    void reserved() { if (integer(4) != 0) { throw std::invalid_argument("nonzero reserved session field"); } }
    step_key key() {
        step_key result;
        block(result.stamp.identity); block(result.stamp.session);
        result.stamp.epoch = integer(8); result.stamp.generation = integer(8);
        result.transaction = integer(8); result.chunk = integer(8); result.position = integer(8);
        result.rows = uint32_t(integer(4)); reserved(); block(result.request_digest);
        return result;
    }
    owner_receipt owner(const step_key & key) {
        owner_receipt result; result.step = key;
        result.rank = uint32_t(integer(4)); reserved(); result.end_position = integer(8);
        block(result.input_digest); block(result.output_digest);
        return result;
    }
    void finish() const {
        if (offset != input.size) { throw std::invalid_argument("trailing session payload bytes"); }
    }
};

void check_control(const header & envelope, kind type, bytes_view payload, const step_key & expected) {
    if (envelope.type != type || envelope.sender > 2 || envelope.receiver > 2 || envelope.sender == envelope.receiver ||
            envelope.session != expected.stamp.session || envelope.epoch != expected.stamp.epoch ||
            envelope.generation != expected.stamp.generation || envelope.transaction != expected.transaction ||
            envelope.chunk != expected.chunk || envelope.position != expected.position || envelope.rows || envelope.hidden ||
            envelope.streams || envelope.element != dtype::none || envelope.fragment || envelope.fragments != 1 ||
            envelope.payload_bytes != payload.size || envelope.total_bytes != payload.size || envelope.sequence == 0) {
        throw std::invalid_argument("session control envelope differs from its pending transaction/body");
    }
}

} // namespace

receipt_wire encode_applied_receipt(const owner_receipt & receipt, const limits & bounds) {
    validate_receipt(receipt, bounds);
    writer<receipt_wire_bytes> encoded;
    encoded.prefix(receipt_kind); encoded.key(receipt.step); encoded.owner(receipt);
    return encoded.finish();
}

commit_wire encode_commit_certificate(const commit_certificate & certificate, const limits & bounds) {
    validate_commit(certificate, bounds);
    writer<commit_wire_bytes> encoded;
    encoded.prefix(commit_kind); encoded.key(certificate.owners[0].step);
    for (const auto & owner : certificate.owners) { encoded.owner(owner); }
    return encoded.finish();
}

digest fingerprint_commit(const commit_certificate & certificate, const limits & bounds) {
    const auto encoded = encode_commit_certificate(certificate, bounds);
    return sha256({{encoded.data(), encoded.size()}});
}

commit_acknowledgement make_commit_ack(const owner_receipt & committed, const commit_certificate & certificate, const limits & bounds) {
    validate_receipt(committed, bounds);
    const auto fingerprint = fingerprint_commit(certificate, bounds);
    if (!(committed == certificate.owners[committed.rank])) {
        throw std::invalid_argument("committed receipt differs from its complete certificate");
    }
    return {committed, fingerprint};
}

commit_ack_wire encode_commit_ack(const commit_acknowledgement & acknowledgement, const limits & bounds) {
    validate_receipt(acknowledgement.owner, bounds);
    if (acknowledgement.certificate_digest == digest{}) { throw std::invalid_argument("missing commit certificate fingerprint"); }
    writer<commit_ack_wire_bytes> encoded;
    encoded.prefix(commit_ack_kind); encoded.key(acknowledgement.owner.step); encoded.owner(acknowledgement.owner);
    encoded.block(acknowledgement.certificate_digest);
    return encoded.finish();
}

completion_wire encode_commit_completion(const commit_certificate & committed, const limits & bounds) {
    const auto fingerprint = fingerprint_commit(committed, bounds);
    writer<completion_wire_bytes> encoded;
    encoded.prefix(completion_kind); encoded.key(committed.owners[0].step); encoded.block(fingerprint);
    return encoded.finish();
}

terminal_wire encode_terminal_outcome(const terminal_outcome & outcome, const limits & bounds) {
    validate_terminal(outcome,bounds);
    writer<terminal_wire_bytes> encoded;
    encoded.prefix(terminal_kind); encoded.key(outcome.step); encoded.block(outcome.certificate_digest);
    encoded.integer(2,4); encoded.integer(outcome.vocabulary,4); encoded.integer(outcome.selected_token,4);
    encoded.integer(float_bits(outcome.selected_logit),4);
    encoded.integer(outcome.mode,4); encoded.integer(outcome.count,4); encoded.integer(outcome.row,4); encoded.integer(0,4);
    for (const auto & candidate : outcome.top) { encoded.integer(candidate.token,4); encoded.integer(float_bits(candidate.logit),4); }
    return encoded.finish();
}

owner_receipt decode_applied_receipt(bytes_view payload, uint32_t authenticated_rank, const step_key & expected, const limits & bounds) {
    if (authenticated_rank > 2) { throw std::invalid_argument("invalid authenticated receipt rank"); }
    reader encoded(payload, receipt_wire_bytes, receipt_kind);
    const auto key = encoded.key();
    const auto receipt = encoded.owner(key);
    encoded.finish();
    validate_receipt(receipt, bounds);
    if (receipt.rank != authenticated_rank || !(key == expected)) {
        throw std::invalid_argument("receipt does not match the authenticated owner and pending transaction");
    }
    return receipt;
}

commit_certificate decode_commit_certificate(bytes_view payload, const step_key & expected, const limits & bounds) {
    reader encoded(payload, commit_wire_bytes, commit_kind);
    const auto key = encoded.key();
    commit_certificate certificate;
    for (auto & owner : certificate.owners) { owner = encoded.owner(key); }
    encoded.finish();
    validate_commit(certificate, bounds);
    if (!(key == expected)) { throw std::invalid_argument("commit does not match the pending transaction"); }
    return certificate;
}

commit_acknowledgement decode_commit_ack(bytes_view payload, uint32_t authenticated_rank, const step_key & expected,
        const digest & expected_certificate, const limits & bounds) {
    if (authenticated_rank > 2 || expected_certificate == digest{}) { throw std::invalid_argument("invalid commit acknowledgement expectation"); }
    reader encoded(payload, commit_ack_wire_bytes, commit_ack_kind);
    const auto key = encoded.key();
    commit_acknowledgement result; result.owner = encoded.owner(key); encoded.block(result.certificate_digest);
    encoded.finish();
    validate_receipt(result.owner, bounds);
    if (result.owner.rank != authenticated_rank || !(key == expected) || result.certificate_digest != expected_certificate) {
        throw std::invalid_argument("commit acknowledgement differs from its authenticated owner/key/certificate");
    }
    return result;
}

owner_receipt decode_applied_ack(const header & authenticated, bytes_view payload, const step_key & expected, const limits & bounds) {
    check_control(authenticated, kind::state_ack, payload, expected);
    if (authenticated.receiver != 0) { throw std::invalid_argument("applied acknowledgement requires coordinator A"); }
    return decode_applied_receipt(payload, authenticated.sender, expected, bounds);
}

commit_certificate decode_commit_decision(const header & authenticated, bytes_view payload, const step_key & expected, const limits & bounds) {
    check_control(authenticated, kind::commit, payload, expected);
    if (authenticated.sender != 0) { throw std::invalid_argument("commit decision requires coordinator A"); }
    return decode_commit_certificate(payload, expected, bounds);
}

commit_acknowledgement decode_committed_ack(const header & authenticated, bytes_view payload, const step_key & expected,
        const digest & expected_certificate, const limits & bounds) {
    check_control(authenticated, kind::commit_ack, payload, expected);
    if (authenticated.receiver != 0) { throw std::invalid_argument("commit acknowledgement requires coordinator A"); }
    return decode_commit_ack(payload, authenticated.sender, expected, expected_certificate, bounds);
}

void decode_commit_completion(const header & authenticated, bytes_view payload, const step_key & expected,
        const digest & expected_certificate, const limits & bounds) {
    check_control(authenticated, kind::commit_complete, payload, expected);
    if (authenticated.sender != 0 || expected_certificate == digest{}) {
        throw std::invalid_argument("global completion requires coordinator A and a local committed certificate");
    }
    reader encoded(payload, completion_wire_bytes, completion_kind);
    const auto key = encoded.key();
    digest fingerprint{}; encoded.block(fingerprint); encoded.finish();
    validate_step(key, bounds);
    if (!(key == expected) || fingerprint != expected_certificate) {
        throw std::invalid_argument("global completion differs from the local committed decision");
    }
}

terminal_outcome decode_terminal_outcome(const header & authenticated, bytes_view payload, const step_key & expected,
        const digest & expected_certificate, uint32_t expected_vocabulary, uint32_t expected_candidates, const limits & bounds) {
    check_control(authenticated,kind::terminal_result,payload,expected);
    if (authenticated.sender != 2 || authenticated.receiver != 0 || expected_certificate == digest{} ||
            !expected_vocabulary || expected_vocabulary > (1u<<20) || !expected_candidates || expected_candidates > 8) {
        throw std::invalid_argument("terminal result requires C-to-A and an explicit pending committed expectation");
    }
    reader encoded(payload,terminal_wire_bytes,terminal_kind);
    terminal_outcome result; result.step = encoded.key(); encoded.block(result.certificate_digest);
    if (encoded.integer(4) != 2) { throw std::invalid_argument("terminal payload requires rank C"); }
    result.vocabulary = uint32_t(encoded.integer(4)); result.selected_token = uint32_t(encoded.integer(4));
    result.selected_logit = from_bits(uint32_t(encoded.integer(4)));
    result.mode = uint32_t(encoded.integer(4)); result.count = uint32_t(encoded.integer(4)); result.row = uint32_t(encoded.integer(4)); encoded.reserved();
    for (auto & candidate : result.top) { candidate.token = uint32_t(encoded.integer(4)); candidate.logit = from_bits(uint32_t(encoded.integer(4))); }
    encoded.finish(); validate_terminal(result,bounds);
    if (!(result.step == expected) || result.certificate_digest != expected_certificate ||
            result.vocabulary != expected_vocabulary || result.count != expected_candidates) {
        throw std::invalid_argument("terminal outcome differs from its pending key/certificate/sampling expectation");
    }
    return result;
}

} // namespace glm_rpc
