#include "stage-sampling.h"

#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace glm_rpc {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "sampling requires IEEE binary32");

uint32_t bits(float value) { uint32_t result; std::memcpy(&result,&value,sizeof(result)); return result; }

float read_float(const uint8_t * bytes) {
    uint32_t encoded = 0;
    for (size_t i = 0; i < 4; ++i) { encoded |= uint32_t(bytes[i]) << (8*i); }
    float result; std::memcpy(&result,&encoded,sizeof(result)); return result;
}

void validate_config(const sampling_config & config) {
    if (!config.vocabulary || config.vocabulary > (1u<<20) || !config.candidates ||
            config.candidates > 8 || config.candidates > config.vocabulary ||
            !std::isfinite(config.temperature) || config.temperature <= 0 ||
            (config.mode != sampling_mode::greedy && config.mode != sampling_mode::temperature_distribution) ||
            (config.mode == sampling_mode::greedy && (config.seed || config.temperature != 1)) ||
            config.seed == LLAMA_DEFAULT_SEED) {
        throw std::invalid_argument("unsupported or noncanonical sampling configuration");
    }
}

llama_sampler_ptr make_sampler(const sampling_config & config) {
    auto params = llama_sampler_chain_default_params(); params.no_perf = true;
    llama_sampler_ptr sampler(llama_sampler_chain_init(params));
    if (!sampler) { throw std::runtime_error("cannot create native sampler chain"); }
    const auto add = [&](llama_sampler * raw) {
        llama_sampler_ptr item(raw);
        if (!item) { throw std::runtime_error("cannot create native sampling policy"); }
        llama_sampler_chain_add(sampler.get(),item.get()); item.release();
    };
    if (config.mode == sampling_mode::greedy) { add(llama_sampler_init_greedy()); }
    else { add(llama_sampler_init_temp(config.temperature)); add(llama_sampler_init_dist(config.seed)); }
    return sampler;
}

} // namespace

digest fingerprint_sampling(const sampling_config & config) {
    validate_config(config);
    std::array<uint8_t,32> encoded{'G','L','M','S','A','M','P',0};
    size_t offset = 8;
    for (const uint32_t value : {1u,uint32_t(config.mode),config.vocabulary,config.seed,bits(config.temperature),config.candidates}) {
        for (size_t i = 0; i < 4; ++i) { encoded[offset++] = uint8_t(value >> (8*i)); }
    }
    return sha256({{encoded.data(),encoded.size()}});
}

struct sampling_transactions::impl {
    sampling_config config;
    limits bounds;
    digest configuration;
    owner_transactions ledger;
    llama_sampler_ptr committed, pending;
    std::optional<sampling_result> applied, last;
    uint64_t draws = 0;

    impl(const sampling_config & config, const epoch_stamp & stamp, const limits & bounds) :
        config(config), bounds(bounds), configuration(fingerprint_sampling(config)), ledger(2,stamp.identity,bounds),
        committed(make_sampler(config)) { ledger.fresh_context(stamp); }

    void invalidate() noexcept {
        ledger.invalidate(); pending.reset(); committed.reset(); applied.reset(); last.reset();
    }

    sampling_result draw(const owner_receipt & receipt, bytes_view payload) {
        validate_receipt(receipt,bounds);
        const uint64_t bytes = uint64_t(receipt.step.rows)*config.vocabulary*4;
        if (receipt.rank != 2 || !payload.data || payload.size != bytes || bytes > max_chunk_bytes ||
                sha256({payload}) != receipt.output_digest) {
            throw std::invalid_argument("sampling requires exact bounded C logit bytes and receipt");
        }
        const auto disposition = ledger.prepare(receipt.step,receipt.input_digest);
        if (disposition == step_disposition::committed || disposition == step_disposition::applied) {
            const auto & cached = disposition == step_disposition::committed ? last : applied;
            if (!cached || !(cached->receipt == receipt)) { throw std::invalid_argument("sampling retry changed its C receipt"); }
            return *cached;
        }
        if (disposition != step_disposition::execute || !committed || pending || applied) {
            throw std::logic_error("sampling has no free pending native state");
        }
        const auto * tail = payload.data+(receipt.step.rows-1)*size_t(config.vocabulary)*4;
        std::vector<llama_token_data> candidates(config.vocabulary);
        for (uint32_t token = 0; token < config.vocabulary; ++token) {
            const float score = read_float(tail+token*4);
            if (!std::isfinite(score) || (config.mode == sampling_mode::temperature_distribution &&
                    !std::isfinite(score/config.temperature))) {
                throw std::invalid_argument("sampling logits or scaled values are nonfinite");
            }
            candidates[token] = {llama_token(token),score,0};
        }
        sampling_result result; result.receipt = receipt; result.configuration = configuration;
        result.ordinal = draws+1; result.count = config.candidates; result.row = receipt.step.rows-1;
        std::array<llama_token_data,8> top{};
        std::partial_sort_copy(candidates.begin(),candidates.end(),top.begin(),top.begin()+result.count,
            [](const auto & a, const auto & b) { return a.logit == b.logit ? a.id < b.id : a.logit > b.logit; });
        for (size_t i = 0; i < result.count; ++i) { result.top[i] = {uint32_t(top[i].id),top[i].logit}; }
        pending.reset(llama_sampler_clone(committed.get()));
        if (!pending) { throw std::runtime_error("cannot clone committed native sampler"); }
        llama_token_data_array values{candidates.data(),candidates.size(),-1,false};
        llama_sampler_apply(pending.get(),&values);
        if (values.selected < 0 || size_t(values.selected) >= values.size || values.data[values.selected].id < 0 ||
                uint32_t(values.data[values.selected].id) >= config.vocabulary) {
            throw std::runtime_error("native sampler returned an invalid selection");
        }
        result.selected_token = uint32_t(values.data[values.selected].id);
        result.selected_logit = read_float(tail+result.selected_token*4);
        llama_sampler_accept(pending.get(),llama_token(result.selected_token));
        if (!(ledger.applied(receipt.step,receipt.output_digest) == receipt)) { throw std::logic_error("sampler owner receipt differs"); }
        applied = result;
        return result;
    }

    sampling_result complete(const commit_certificate & certificate) {
        const auto fingerprint = fingerprint_commit(certificate,bounds);
        const auto receipt = ledger.commit(certificate);
        if (applied && applied->receipt == receipt) {
            if (!pending) { throw std::logic_error("missing pending native sampler state"); }
            committed = std::move(pending);
            applied->certificate_digest = fingerprint;
            last = applied; applied.reset(); ++draws;
            return *last;
        }
        if (!last || !(last->receipt == receipt) || last->certificate_digest != fingerprint) {
            throw std::invalid_argument("completion changed the cached sampling decision");
        }
        return *last;
    }
};

sampling_transactions::sampling_transactions(const sampling_config & config, const epoch_stamp & stamp, const limits & bounds) :
    state(std::make_unique<impl>(config,stamp,bounds)) {}
sampling_transactions::~sampling_transactions() = default;

sampling_result sampling_transactions::draw(const owner_receipt & terminal, bytes_view logits) {
    try { return state->draw(terminal,logits); }
    catch (...) { state->invalidate(); throw; }
}

sampling_result sampling_transactions::complete_after_global(const commit_certificate & certificate) {
    try { return state->complete(certificate); }
    catch (...) { state->invalidate(); throw; }
}

void sampling_transactions::invalidate() noexcept { state->invalidate(); }
uint64_t sampling_transactions::committed_draws() const { return state->draws; }

} // namespace glm_rpc
