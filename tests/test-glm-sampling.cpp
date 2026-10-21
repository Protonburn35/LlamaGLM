#include "stage-sampling.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace glm_rpc;

static void require(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }
template<class F> static void refused(F function) {
    try { function(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid sampling operation was accepted");
}
static digest hash(const std::string & value) { return sha256({{reinterpret_cast<const uint8_t *>(value.data()),value.size()}}); }
static std::string hex(const digest & value) {
    std::string result;
    for (auto byte : value) { result += "0123456789abcdef"[byte>>4]; result += "0123456789abcdef"[byte&15]; }
    return result;
}
static uint32_t bits(float value) { uint32_t result; std::memcpy(&result,&value,sizeof(result)); return result; }
static limits bounds() { limits value; value.hidden = 128; value.rows = 64; value.context = 256; return value; }
static epoch_stamp stamp(uint64_t generation = 1) {
    epoch_stamp result{hash("native sampling fixture"),{},generation,generation}; result.session[0] = uint8_t(generation); return result;
}
static sampling_config config(bool random = false, uint32_t vocabulary = 128) {
    return {random ? sampling_mode::temperature_distribution : sampling_mode::greedy,vocabulary,
        random ? 53u : 0u,random ? 0.75f : 1.0f,std::min(8u,vocabulary)};
}

struct logits {
    std::vector<uint8_t> bytes;
    std::vector<float> tail;
    commit_certificate certificate;
    logits(uint64_t id, uint64_t position, uint32_t rows = 1, uint32_t vocabulary = 128, uint64_t generation = 1) {
        tail.resize(vocabulary); bytes.resize(size_t(rows)*vocabulary*4);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t token = 0; token < vocabulary; ++token) {
                const float score = float((token*7+row*11+id*3)%31)/64;
                const auto encoded = bits(score);
                for (size_t i = 0; i < 4; ++i) { bytes[(size_t(row)*vocabulary+token)*4+i] = uint8_t(encoded>>(8*i)); }
                if (row == rows-1) { tail[token] = score; }
            }
        }
        step_key step{stamp(generation),id,id,position,rows,hash("input tokens "+std::to_string(id))};
        auto input = step.request_digest;
        for (uint32_t rank = 0; rank < 3; ++rank) {
            const auto output = rank == 2 ? sha256({view()}) : hash("boundary "+std::to_string(rank)+"/"+std::to_string(id));
            certificate.owners[rank] = {step,rank,position+rows,input,output}; input = output;
        }
    }
    bytes_view view() const { return {bytes.data(),bytes.size()}; }
    const owner_receipt & receipt() const { return certificate.owners[2]; }
    void rehash() { certificate.owners[2].output_digest = sha256({view()}); }
};

static void same_selection(const sampling_result & a, const sampling_result & b) {
    require(a.receipt == b.receipt && a.configuration == b.configuration && a.ordinal == b.ordinal &&
            a.selected_token == b.selected_token && bits(a.selected_logit) == bits(b.selected_logit) &&
            a.count == b.count && a.row == b.row,"sampling retry changed selection facts");
    for (size_t i = 0; i < a.top.size(); ++i) {
        require(a.top[i].token == b.top[i].token && bits(a.top[i].logit) == bits(b.top[i].logit),"sampling retry changed raw top candidates");
    }
}

static void configuration() {
    require(hex(fingerprint_sampling(config())) == "a843c035a6e93741d4b2536f9b3e4c6a38e898f07451599f94e1c5b68691abbd" &&
            hex(fingerprint_sampling(config(true))) == "941c73dfe39e15e4015767934ba7d9111a8decf21739319f9c358e11b8ff2234",
            "sampling configuration differs from independent Python LE/SHA256 goldens");
    const auto original = fingerprint_sampling(config(true));
    for (int field = 0; field < 4; ++field) {
        auto changed = config(true);
        if (field == 0) { ++changed.seed; }
        if (field == 1) { changed.temperature = 0.8f; }
        if (field == 2) { ++changed.vocabulary; }
        if (field == 3) { --changed.candidates; }
        require(fingerprint_sampling(changed) != original,"sampling configuration field was not bound");
    }
    for (int failure = 0; failure < 11; ++failure) {
        auto invalid = config();
        if (failure == 0) { invalid.mode = sampling_mode(0); }
        if (failure == 1) { invalid.seed = LLAMA_DEFAULT_SEED; }
        if (failure == 2) { invalid.seed = 53; }
        if (failure == 3) { invalid.temperature = 0.75f; }
        if (failure == 4) { invalid.vocabulary = 0; }
        if (failure == 5) { invalid.vocabulary = (1u<<20)+1; }
        if (failure == 6) { invalid.candidates = 0; }
        if (failure == 7) { invalid.candidates = 9; }
        if (failure == 8) { invalid.temperature = std::numeric_limits<float>::quiet_NaN(); }
        if (failure == 9) { invalid = config(true); invalid.temperature = 0; }
        if (failure == 10) { invalid = config(true); invalid.seed = LLAMA_DEFAULT_SEED; }
        refused([&] { sampling_transactions sampler(invalid,stamp(),bounds()); });
    }
}

static void greedy() {
    sampling_transactions sampler(config(),stamp(),bounds());
    logits first(1,0,17);
    const auto pending = sampler.draw(first.receipt(),first.view());
    const auto selected = uint32_t(std::max_element(first.tail.begin(),first.tail.end())-first.tail.begin());
    require(pending.selected_token == selected && bits(pending.selected_logit) == bits(first.tail[selected]) &&
            pending.row == 16 && pending.ordinal == 1 && pending.certificate_digest == digest{} && !sampler.committed_draws(),
            "greedy draw published state or selected the wrong row/token");
    same_selection(pending,sampler.draw(first.receipt(),first.view()));
    auto committed = sampler.complete_after_global(first.certificate);
    same_selection(pending,committed);
    require(committed.certificate_digest == fingerprint_commit(first.certificate,bounds()) && sampler.committed_draws() == 1,
            "greedy completion lost the certificate or advanced more than once");
    same_selection(committed,sampler.complete_after_global(first.certificate));
    logits second(2,17);
    const auto next = sampler.draw(second.receipt(),second.view());
    same_selection(committed,sampler.draw(first.receipt(),first.view()));
    same_selection(committed,sampler.complete_after_global(first.certificate));
    same_selection(next,sampler.draw(second.receipt(),second.view()));
    sampler.complete_after_global(second.certificate);
    require(sampler.committed_draws() == 2,"late greedy retry displaced pending state");
}

static void distribution() {
    const auto settings = config(true);
    sampling_transactions sampler(settings,stamp(),bounds());
    llama_sampler_ptr reference(llama_sampler_init_dist(settings.seed));
    llama_sampler_ptr temperature(llama_sampler_init_temp(settings.temperature));
    size_t outside_diagnostics = 0;
    std::optional<logits> previous;
    std::optional<sampling_result> last;
    for (uint64_t id = 1; id <= 64; ++id) {
        logits input(id,id-1);
        std::vector<llama_token_data> full(settings.vocabulary);
        for (size_t i = 0; i < full.size(); ++i) { full[i] = {llama_token(i),input.tail[i],0}; }
        llama_token_data_array direct{full.data(),full.size(),-1,false};
        llama_sampler_apply(temperature.get(),&direct); llama_sampler_apply(reference.get(),&direct);
        const auto expected = uint32_t(direct.data[direct.selected].id);
        llama_sampler_accept(reference.get(),llama_token(expected));
        const auto pending = sampler.draw(input.receipt(),input.view());
        require(pending.selected_token == expected && pending.ordinal == id && sampler.committed_draws() == id-1 &&
                bits(pending.selected_logit) == bits(input.tail[expected]),"transactional native full-vocabulary draw differs from direct reference");
        same_selection(pending,sampler.draw(input.receipt(),input.view()));
        if (previous) {
            same_selection(*last,sampler.draw(previous->receipt(),previous->view()));
            same_selection(*last,sampler.complete_after_global(previous->certificate));
        }
        if (std::none_of(pending.top.begin(),pending.top.begin()+pending.count,
                [&](const auto & item) { return item.token == expected; })) { ++outside_diagnostics; }
        last = sampler.complete_after_global(input.certificate);
        same_selection(pending,*last);
        same_selection(*last,sampler.complete_after_global(input.certificate));
        require(sampler.committed_draws() == id,"retry consumed extra RNG state");
        previous = std::move(input);
    }
    require(outside_diagnostics > 32,"diagnostic top-eight accidentally restricted the sampling distribution");
}

static void failures() {
    for (int failure = 0; failure < 8; ++failure) {
        sampling_transactions sampler(config(true),stamp(),bounds());
        logits input(1,0);
        if (failure == 0) { input.bytes[0] ^= 1; }
        if (failure == 1) { input.certificate.owners[2].rank = 1; }
        if (failure == 2) { input.certificate.owners[2].step.stamp.session[1] = 1; }
        if (failure == 3) { input.bytes.pop_back(); }
        if (failure <= 3) { refused([&] { sampler.draw(input.receipt(),input.view()); }); }
        else {
            sampler.draw(input.receipt(),input.view());
            if (failure == 4) { input.certificate.owners[2].output_digest = hash("other logits"); }
            if (failure == 5) { input.certificate.owners[1].output_digest = hash("other boundary"); }
            if (failure == 6) { ++input.certificate.owners[0].end_position; }
            if (failure == 7) { ++input.certificate.owners[2].step.transaction; }
            refused([&] { sampler.complete_after_global(input.certificate); });
        }
        require(!sampler.committed_draws(),"failed draw/completion advanced committed RNG");
        logits valid(1,0);
        refused([&] { sampler.draw(valid.receipt(),valid.view()); });
    }
    sampling_transactions sampler(config(true),stamp(),bounds());
    logits first(1,0); sampler.draw(first.receipt(),first.view()); sampler.complete_after_global(first.certificate);
    auto changed = first.certificate;
    changed.owners[0].output_digest = changed.owners[1].input_digest = hash("changed complete certificate");
    refused([&] { sampler.complete_after_global(changed); });
    require(sampler.committed_draws() == 1,"conflicting old certificate advanced RNG");
}

static void replay() {
    sampling_transactions failed(config(true),stamp(),bounds());
    std::vector<uint32_t> expected;
    for (uint64_t id = 1; id <= 6; ++id) {
        logits input(id,id-1);
        expected.push_back(failed.draw(input.receipt(),input.view()).selected_token);
        if (id != 6) { failed.complete_after_global(input.certificate); }
    }
    failed.invalidate();
    require(failed.committed_draws() == 5,"unpublished pending draw became committed on invalidation");
    sampling_transactions fresh(config(true),stamp(2),bounds());
    for (uint64_t id = 1; id <= 6; ++id) {
        logits input(id,id-1,1,128,2);
        const auto result = fresh.draw(input.receipt(),input.view());
        require(result.selected_token == expected[id-1],"new-epoch replay did not restore native seed/RNG progression");
        fresh.complete_after_global(input.certificate);
    }
    require(fresh.committed_draws() == 6,"replayed sampler has the wrong RNG cursor");
}

static void limits_and_values() {
    for (uint32_t encoded : {0x7fc00000u,0x7f800000u,0xff800000u}) {
        sampling_transactions sampler(config(true),stamp(),bounds()); logits input(1,0);
        for (size_t i = 0; i < 4; ++i) { input.bytes[i] = uint8_t(encoded>>(8*i)); } input.rehash();
        refused([&] { sampler.draw(input.receipt(),input.view()); });
    }
    auto tiny = config(true); tiny.temperature = std::numeric_limits<float>::min();
    sampling_transactions overflow(tiny,stamp(),bounds()); logits large(1,0);
    const auto encoded = bits(16.0f);
    for (size_t i = 0; i < 4; ++i) { large.bytes[i] = uint8_t(encoded>>(8*i)); } large.rehash();
    refused([&] { overflow.draw(large.receipt(),large.view()); });
    sampling_transactions null_input(config(),stamp(),bounds()); logits input(1,0);
    refused([&] { null_input.draw(input.receipt(),{nullptr,input.bytes.size()}); });
    sampling_transactions huge(config(true,1u<<20),stamp(),bounds());
    auto receipt = input.receipt(); receipt.step.rows = 64; receipt.end_position = 64;
    uint8_t byte = 0;
    refused([&] { huge.draw(receipt,{&byte,size_t(64)*(1u<<20)*4}); });
    sampling_transactions singleton(config(true,1),stamp(),bounds()); logits one(1,0,1,1);
    require(singleton.draw(one.receipt(),one.view()).selected_token == 0,"single-token vocabulary failed");
    singleton.complete_after_global(one.certificate);
}

int main() {
    try {
        configuration(); greedy(); distribution(); failures(); replay(); limits_and_values();
        std::puts("GLM sampling: six suites passed (native full-vocabulary RNG, clone isolation, commit/retries, fail-closed conflicts, reset replay); stochastic wire/production serving NOT_IMPLEMENTED");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"GLM sampling test failed: %s\n",error.what()); return 1;
    }
}
