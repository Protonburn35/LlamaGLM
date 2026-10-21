#include "stage-sampling.h"
#include "stage-sampling-codec.h"
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

static header control(kind type, uint32_t from, uint32_t to, size_t bytes, const step_key * step = nullptr) {
    header result; result.type = type; result.sender = from; result.receiver = to;
    const auto epoch = step ? step->stamp : stamp();
    result.session = epoch.session; result.epoch = epoch.epoch; result.generation = epoch.generation;
    result.transaction = step ? step->transaction : 0; result.chunk = step ? step->chunk : 0; result.position = step ? step->position : 0;
    result.payload_bytes = uint32_t(bytes); result.total_bytes = bytes; result.sequence = 1; return result;
}

static void agreement_codec() {
    const auto cfg = config(true);
    const auto proposal = encode_sampling_setup(cfg,stamp(),bounds());
    const auto acknowledgement = encode_sampling_setup_ack(cfg,stamp(),bounds());
    require(hex(sha256({{proposal.data(),proposal.size()}})) == "e1983492ee23cbba6c3ba033e5041ff31babe0e305b771798ca9131df6d1c660" &&
            hex(sha256({{acknowledgement.data(),acknowledgement.size()}})) == "ec3dd9af534f7067d67f44c46411749795460f974778c9404ec03ae45ef464f4",
            "sampling agreement differs from independent Python bytes");
    for (uint32_t rank : {1u,2u}) {
        auto request = control(kind::sampling_setup,0,rank,proposal.size());
        auto ack = control(kind::sampling_setup_ack,rank,0,acknowledgement.size());
        decode_sampling_setup(request,{proposal.data(),proposal.size()},rank,cfg,stamp(),bounds());
        decode_sampling_setup_ack(ack,{acknowledgement.data(),acknowledgement.size()},rank,cfg,stamp(),bounds());
        for (size_t size = 0; size < proposal.size(); ++size) {
            request.payload_bytes = uint32_t(size); request.total_bytes = size;
            refused([&] { decode_sampling_setup(request,{proposal.data(),size},rank,cfg,stamp(),bounds()); });
        }
        request = control(kind::sampling_setup,0,rank,proposal.size());
        for (size_t offset = 0; offset < proposal.size(); ++offset) {
            auto changed = proposal; changed[offset] ^= 1;
            refused([&] { decode_sampling_setup(request,{changed.data(),changed.size()},rank,cfg,stamp(),bounds()); });
        }
        refused([&] { decode_sampling_setup(request,{acknowledgement.data(),acknowledgement.size()},rank,cfg,stamp(),bounds()); });
        auto changed = cfg; ++changed.seed;
        refused([&] { decode_sampling_setup_ack(ack,{acknowledgement.data(),acknowledgement.size()},rank,changed,stamp(),bounds()); });
        request.sender = rank; request.receiver = 0;
        refused([&] { decode_sampling_setup(request,{proposal.data(),proposal.size()},rank,cfg,stamp(),bounds()); });
    }
}

static sampling_result codec_result(logits & input) {
    sampling_transactions sampler(config(true),stamp(),bounds());
    sampler.draw(input.receipt(),input.view());
    auto result = sampler.complete_after_global(input.certificate);
    result.selected_token = 0; result.selected_logit = input.tail[0];
    return result;
}

static void result_codec() {
    logits input(1,0); const auto result = codec_result(input); const auto cfg = config(true);
    const auto encoded = encode_sampling_result(result,cfg,input.certificate,bounds());
    require(hex(sha256({{encoded.data(),encoded.size()}})) == "1fdca30587e866b0eb81066221cbee1ecbe35bc787c87ac9fd8edbe0043c711f",
            "sampled result differs from independent Python bytes");
    auto frame = control(kind::sampled_result,2,0,encoded.size(),&input.receipt().step);
    same_selection(result,decode_sampling_result(frame,{encoded.data(),encoded.size()},cfg,input.certificate,1,bounds()));
    for (size_t size = 0; size < encoded.size(); ++size) {
        frame.payload_bytes = uint32_t(size); frame.total_bytes = size;
        refused([&] { decode_sampling_result(frame,{encoded.data(),size},cfg,input.certificate,1,bounds()); });
    }
    frame = control(kind::sampled_result,2,0,encoded.size(),&input.receipt().step);
    for (size_t offset : {0u,8u,10u,12u,16u,48u,64u,72u,80u,88u,96u,104u,108u,112u,144u,176u,208u,216u,220u,232u,236u,240u,244u,312u}) {
        auto changed = encoded; changed[offset] ^= 1;
        refused([&] { decode_sampling_result(frame,{changed.data(),changed.size()},cfg,input.certificate,1,bounds()); });
    }
    for (int failure = 0; failure < 6; ++failure) {
        auto changed = result;
        if (failure == 0) { changed.selected_token = 128; }
        if (failure == 1) { changed.selected_logit = 1; }
        if (failure == 2) { changed.top[1] = changed.top[0]; }
        if (failure == 3) { changed.top[0].logit = std::numeric_limits<float>::quiet_NaN(); }
        if (failure == 4) { std::swap(changed.top[0],changed.top[1]); }
        if (failure == 5) { changed.certificate_digest = {}; }
        refused([&] { encode_sampling_result(changed,cfg,input.certificate,bounds()); });
    }
    refused([&] { decode_sampling_result(frame,{encoded.data(),encoded.size()},cfg,input.certificate,2,bounds()); });
    auto changed = cfg; changed.temperature = 0.8f;
    refused([&] { decode_sampling_result(frame,{encoded.data(),encoded.size()},changed,input.certificate,1,bounds()); });
    frame.type = kind::terminal_result;
    refused([&] { decode_sampling_result(frame,{encoded.data(),encoded.size()},cfg,input.certificate,1,bounds()); });
    frame.type = kind::sampled_result;
    uint32_t mutation = 53;
    for (size_t i = 0; i < 2000; ++i) {
        mutation = mutation*1664525u+1013904223u;
        auto changed_bytes = encoded;
        changed_bytes[mutation%changed_bytes.size()] ^= uint8_t(1u<<((mutation>>16)%8));
        std::optional<sampling_result> accepted;
        try { accepted = decode_sampling_result(frame,{changed_bytes.data(),changed_bytes.size()},cfg,input.certificate,1,bounds()); }
        catch (const std::invalid_argument &) {}
        if (accepted) { require(encode_sampling_result(*accepted,cfg,input.certificate,bounds()) == changed_bytes,"accepted mutation was not canonical"); }
    }
    auto narrow = cfg; narrow.candidates = 4;
    sampling_transactions sampler(narrow,stamp(),bounds()); sampler.draw(input.receipt(),input.view());
    auto limited = sampler.complete_after_global(input.certificate);
    encode_sampling_result(limited,narrow,input.certificate,bounds());
    limited.top[4] = limited.top[0];
    refused([&] { encode_sampling_result(limited,narrow,input.certificate,bounds()); });
}

static void authenticated_sampling() {
    logits input(1,0); const auto result = codec_result(input); const auto cfg = config(true);
    const auto payload = encode_sampling_result(result,cfg,input.certificate,bounds());
    secret psk{}; psk[0] = 53;
    binding peer; peer.server_nonce[0] = 1; peer.client_nonce[0] = 2; peer.identity = stamp().identity;
    peer.sender = 2; peer.receiver = 0; peer.link = channel::control;
    const auto key = derive_link_key(psk,peer);
    sender tx(key,peer,bounds()); receiver rx(key,peer,bounds());
    const auto packet = tx.encode(control(kind::sampled_result,2,0,payload.size(),&input.receipt().step),{payload.data(),payload.size()});
    wire_header raw{}; std::copy_n(packet.begin(),raw.size(),raw.begin());
    const bytes_view body{packet.data()+raw.size(),packet.size()-raw.size()};
    const auto authenticated = rx.authenticate(raw,body);
    same_selection(result,decode_sampling_result(authenticated,body,cfg,input.certificate,1,bounds()));
    auto changed = packet; changed.back() ^= 1;
    receiver fresh(key,peer,bounds());
    refused([&] { fresh.authenticate(raw,{changed.data()+raw.size(),changed.size()-raw.size()}); });
    refused([&] { fresh.authenticate(raw,body); });
    auto data_peer = peer; data_peer.link = channel::data;
    sender wrong_channel(derive_link_key(psk,data_peer),data_peer,bounds());
    refused([&] { wrong_channel.encode(control(kind::sampled_result,2,0,payload.size(),&input.receipt().step),{payload.data(),payload.size()}); });
}

int main() {
    try {
        configuration(); greedy(); distribution(); failures(); replay(); limits_and_values();
        agreement_codec(); result_codec(); authenticated_sampling();
        std::puts("GLM sampling: nine suites passed (native RNG/clone/retries/replay, canonical agreement/results, authenticated roles/configuration); production serving NOT_IMPLEMENTED");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"GLM sampling test failed: %s\n",error.what()); return 1;
    }
}
