#include "stage-sampling-codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace glm_rpc {
namespace {

constexpr std::array<uint8_t,8> domain{'G','L','M','S','R','P','C',0};
constexpr uint16_t result_kind = 7, setup_kind = 8, setup_ack_kind = 9;
uint32_t bits(float value) { uint32_t result; std::memcpy(&result,&value,sizeof(result)); return result; }
float floating(uint32_t value) { float result; std::memcpy(&result,&value,sizeof(result)); return result; }

template<size_t N> struct writer {
    std::array<uint8_t,N> value{}; size_t offset = 0;
    void integer(uint64_t number, size_t count) {
        if (count > 8 || offset > N || count > N-offset) { throw std::logic_error("sampling encoder capacity exceeded"); }
        for (size_t i = 0; i < count; ++i) { value[offset++] = uint8_t(number>>(8*i)); }
    }
    template<size_t M> void block(const std::array<uint8_t,M> & bytes) { for (auto byte : bytes) { integer(byte,1); } }
    void prefix(uint16_t kind) { block(domain); integer(1,2); integer(kind,2); integer(0,4); }
    void stamp(const epoch_stamp & stamp) { block(stamp.identity); block(stamp.session); integer(stamp.epoch,8); integer(stamp.generation,8); }
    void key(const step_key & key) {
        stamp(key.stamp); integer(key.transaction,8); integer(key.chunk,8); integer(key.position,8);
        integer(key.rows,4); integer(0,4); block(key.request_digest);
    }
    std::array<uint8_t,N> finish() { if (offset != N) { throw std::logic_error("sampling encoder size differs"); } return value; }
};

struct reader {
    bytes_view bytes; size_t offset = 0;
    uint64_t integer(size_t count) {
        if (count > 8 || offset > bytes.size || count > bytes.size-offset) { throw std::invalid_argument("truncated sampling result"); }
        uint64_t value = 0;
        for (size_t i = 0; i < count; ++i) { value |= uint64_t(bytes.data[offset++])<<(8*i); }
        return value;
    }
    template<size_t N> void block(std::array<uint8_t,N> & value) { for (auto & byte : value) { byte = uint8_t(integer(1)); } }
    explicit reader(bytes_view payload) : bytes(payload) {
        if (!bytes.data || bytes.size != sampled_wire_bytes) { throw std::invalid_argument("sampling result requires its exact length"); }
        std::array<uint8_t,8> magic{}; block(magic);
        if (magic != domain || integer(2) != 1 || integer(2) != result_kind || integer(4)) {
            throw std::invalid_argument("unknown sampling result domain/version/kind");
        }
    }
    step_key key() {
        step_key result; block(result.stamp.identity); block(result.stamp.session);
        result.stamp.epoch = integer(8); result.stamp.generation = integer(8);
        result.transaction = integer(8); result.chunk = integer(8); result.position = integer(8); result.rows = uint32_t(integer(4));
        if (integer(4)) { throw std::invalid_argument("nonzero sampling key reserved field"); }
        block(result.request_digest); return result;
    }
};

void envelope(const header & frame, bytes_view payload, kind type, uint32_t sender, uint32_t receiver,
        const epoch_stamp & stamp, uint64_t transaction = 0, uint64_t chunk = 0, uint64_t position = 0) {
    if (frame.type != type || sender > 2 || receiver > 2 || sender == receiver || frame.sender != sender || frame.receiver != receiver ||
            frame.session != stamp.session || frame.epoch != stamp.epoch || frame.generation != stamp.generation ||
            frame.transaction != transaction || frame.chunk != chunk || frame.position != position || frame.rows || frame.hidden ||
            frame.streams || frame.element != dtype::none || frame.fragment || frame.fragments != 1 || !frame.sequence ||
            frame.payload_bytes != payload.size || frame.total_bytes != payload.size) {
        throw std::invalid_argument("sampling control envelope differs from local state/authority");
    }
}

sampling_setup_wire setup(const sampling_config & config, const epoch_stamp & stamp, const limits & bounds, bool acknowledgement) {
    validate_step({stamp,1,1,0,1,fingerprint_sampling(config)},bounds);
    writer<sampling_setup_wire_bytes> encoded;
    encoded.prefix(acknowledgement ? setup_ack_kind : setup_kind); encoded.stamp(stamp); encoded.block(canonical_sampling(config));
    return encoded.finish();
}

void validate(const sampling_result & result, const sampling_config & config, const commit_certificate & certificate, const limits & bounds) {
    if (result.configuration != fingerprint_sampling(config) || result.certificate_digest != fingerprint_commit(certificate,bounds) ||
            !(result.receipt == certificate.owners[2]) || result.receipt.rank != 2 || !result.ordinal || result.ordinal > bounds.context ||
            result.selected_token >= config.vocabulary || !std::isfinite(result.selected_logit) ||
            result.count != config.candidates || result.row != result.receipt.step.rows-1) {
        throw std::invalid_argument("sampled outcome differs from local configuration/committed C receipt");
    }
    bool found = false;
    for (size_t i = 0; i < result.top.size(); ++i) {
        const auto & candidate = result.top[i];
        if (i >= result.count) {
            if (candidate.token || bits(candidate.logit)) { throw std::invalid_argument("nonzero unused sampled diagnostic"); }
            continue;
        }
        if (candidate.token >= config.vocabulary || !std::isfinite(candidate.logit) || (i &&
                (candidate.logit > result.top[i-1].logit || (candidate.logit == result.top[i-1].logit && candidate.token <= result.top[i-1].token)))) {
            throw std::invalid_argument("invalid sampled diagnostic order/value");
        }
        for (size_t j = 0; j < i; ++j) { if (candidate.token == result.top[j].token) { throw std::invalid_argument("duplicate sampled diagnostic"); } }
        if (candidate.token == result.selected_token) {
            if (bits(candidate.logit) != bits(result.selected_logit)) { throw std::invalid_argument("sampled token/raw score differs"); }
            found = true;
        }
    }
    const auto & last = result.top[result.count-1];
    if ((!found && (result.count == config.vocabulary || result.selected_logit > last.logit ||
            (result.selected_logit == last.logit && result.selected_token < last.token))) ||
            (config.mode == sampling_mode::greedy && result.selected_token != result.top[0].token)) {
        throw std::invalid_argument("sampled token conflicts with canonical raw diagnostics/policy");
    }
}

} // namespace

sampling_setup_wire encode_sampling_setup(const sampling_config & config, const epoch_stamp & stamp, const limits & bounds) {
    return setup(config,stamp,bounds,false);
}
sampling_setup_wire encode_sampling_setup_ack(const sampling_config & config, const epoch_stamp & stamp, const limits & bounds) {
    return setup(config,stamp,bounds,true);
}
void decode_sampling_setup(const header & frame, bytes_view payload, uint32_t rank,
        const sampling_config & config, const epoch_stamp & stamp, const limits & bounds) {
    envelope(frame,payload,kind::sampling_setup,0,rank,stamp);
    const auto expected = encode_sampling_setup(config,stamp,bounds);
    if (!payload.data || payload.size != expected.size() || !std::equal(expected.begin(),expected.end(),payload.data)) {
        throw std::invalid_argument("sampling setup differs from exact local configuration");
    }
}
void decode_sampling_setup_ack(const header & frame, bytes_view payload, uint32_t rank,
        const sampling_config & config, const epoch_stamp & stamp, const limits & bounds) {
    envelope(frame,payload,kind::sampling_setup_ack,rank,0,stamp);
    const auto expected = encode_sampling_setup_ack(config,stamp,bounds);
    if (!payload.data || payload.size != expected.size() || !std::equal(expected.begin(),expected.end(),payload.data)) {
        throw std::invalid_argument("sampling setup acknowledgement differs from exact local agreement");
    }
}

sampled_wire encode_sampling_result(const sampling_result & result, const sampling_config & config,
        const commit_certificate & certificate, const limits & bounds) {
    validate(result,config,certificate,bounds);
    writer<sampled_wire_bytes> encoded; encoded.prefix(result_kind); encoded.key(result.receipt.step);
    encoded.block(result.certificate_digest); encoded.block(result.configuration); encoded.integer(result.ordinal,8);
    for (uint32_t value : {2u,config.vocabulary,result.selected_token,bits(result.selected_logit),uint32_t(config.mode),result.count,result.row,0u}) {
        encoded.integer(value,4);
    }
    for (const auto & candidate : result.top) { encoded.integer(candidate.token,4); encoded.integer(bits(candidate.logit),4); }
    encoded.integer(0,8); return encoded.finish();
}

sampling_result decode_sampling_result(const header & frame, bytes_view payload, const sampling_config & config,
        const commit_certificate & certificate, uint64_t ordinal, const limits & bounds) {
    const auto & expected = certificate.owners[2];
    envelope(frame,payload,kind::sampled_result,2,0,expected.step.stamp,expected.step.transaction,expected.step.chunk,expected.step.position);
    reader encoded(payload); const auto key = encoded.key();
    sampling_result result; result.receipt = expected; encoded.block(result.certificate_digest); encoded.block(result.configuration);
    result.ordinal = encoded.integer(8);
    if (encoded.integer(4) != 2 || encoded.integer(4) != config.vocabulary) { throw std::invalid_argument("sampled result owner/vocabulary differs"); }
    result.selected_token = uint32_t(encoded.integer(4)); result.selected_logit = floating(uint32_t(encoded.integer(4)));
    if (encoded.integer(4) != uint32_t(config.mode)) { throw std::invalid_argument("sampled result policy differs"); }
    result.count = uint32_t(encoded.integer(4)); result.row = uint32_t(encoded.integer(4));
    if (encoded.integer(4)) { throw std::invalid_argument("nonzero sampled result reserved field"); }
    for (auto & candidate : result.top) { candidate.token = uint32_t(encoded.integer(4)); candidate.logit = floating(uint32_t(encoded.integer(4))); }
    if (encoded.integer(8) || encoded.offset != payload.size || !(key == expected.step) || result.ordinal != ordinal) {
        throw std::invalid_argument("sampled result coordinates/ordinal/reserved fields differ");
    }
    validate(result,config,certificate,bounds); return result;
}

} // namespace glm_rpc
