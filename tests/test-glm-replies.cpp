#include "stage-replies.h"
#include "stage-locked-pages.h"
#include "glm-fixture-lock-quota.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <type_traits>

using namespace glm_rpc;

static void require(bool condition, const char * message) { if (!condition) { throw std::runtime_error(message); } }
template<class F> static void refused(F function) {
    try { function(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid reply operation succeeded");
}
static digest hash(bytes_view bytes) { return sha256({bytes}); }
static limits bounds() { limits value; value.hidden = 128; value.rows = 64; value.context = 256; return value; }
static epoch_stamp stamp() { epoch_stamp value; value.identity[0] = 53; value.session[0] = 1; value.epoch = value.generation = 1; return value; }
static reply_limits pageable(uint64_t bytes = 4096, uint32_t entries = 4) { return {bytes,0,entries,reply_memory::pageable}; }

struct step_fixture {
    std::array<std::vector<uint8_t>, 3> outputs;
    commit_certificate certificate;
    step_fixture(uint64_t transaction = 1, size_t bytes = 64) {
        digest input{}; input[0] = uint8_t(transaction);
        const step_key key{stamp(),transaction,transaction,transaction-1,1,input};
        for (uint32_t rank = 0; rank < 3; ++rank) {
            outputs[rank].resize(bytes);
            for (size_t i = 0; i < bytes; ++i) { outputs[rank][i] = uint8_t(i + rank + transaction); }
            certificate.owners[rank] = {key,rank,transaction,input,hash(payload(rank))};
            input = certificate.owners[rank].output_digest;
        }
        validate_commit(certificate, bounds());
    }
    bytes_view payload(uint32_t rank = 0) const { return {outputs[rank].data(), outputs[rank].size()}; }
};

static void lifecycle() {
    reply_cache cache(0, stamp(), bounds(), pageable());
    step_fixture first;
    auto retained = cache.retain_applied(first.certificate.owners[0], first.payload());
    require(cache.applied_reply(first.certificate.owners[0].step) == retained &&
            cache.retain_applied(first.certificate.owners[0], first.payload()) == retained, "applied retry lost retained bytes");
    const auto expected = first.certificate.owners[0].output_digest;
    first.outputs[0].assign(64, 0);
    require(hash(retained->payload()) == expected && !retained->pages_locked(), "reply aliases borrowed graph bytes");
    refused([&] { retained->verify_locks(); });
    refused([&] { cache.committed_reply(first.certificate.owners[0].step); });
    require(cache.commit(first.certificate) == retained && cache.commit(first.certificate) == retained, "local commit lost its reply");
    step_fixture second(2);
    const auto next = cache.retain_applied(second.certificate.owners[0], second.payload());
    require(cache.committed_reply(first.certificate.owners[0].step) == retained, "pending step evicted the previous retry");
    cache.commit(second.certificate);
    refused([&] { cache.committed_reply(first.certificate.owners[0].step); });
    refused([&] { cache.commit(first.certificate); });
    require(hash(retained->payload()) == expected && cache.committed_reply(second.certificate.owners[0].step) == next &&
            cache.stats().payload_bytes == 128, "retired lease lost ownership or accounting");
    retained.reset();
    require(cache.stats().payload_bytes == 64 && cache.stats().entries == 1, "retired reply charge outlived final lease");
}

static void capacities() {
    reply_cache cache(0, stamp(), bounds(), pageable(128,2));
    step_fixture first, second(2), third(3);
    auto retired = cache.retain_applied(first.certificate.owners[0], first.payload()); cache.commit(first.certificate);
    cache.retain_applied(second.certificate.owners[0], second.payload()); cache.commit(second.certificate);
    refused([&] { cache.retain_applied(third.certificate.owners[0], third.payload()); });
    require(cache.stats().payload_bytes == 128 && cache.stats().entries == 2 && cache.stats().refusals == 1,
            "retained old reply escaped capacity or failed reservation leaked");
    retired.reset();
    cache.retain_applied(third.certificate.owners[0], third.payload());
    require(cache.stats().peak_payload_bytes == 128 && cache.stats().entries == 2, "reply reservation exceeded its shared cap");
    cache.commit(third.certificate);
    require(cache.stats().payload_bytes == 64, "committing next reply did not retire internal old ownership");
}

static void malformed() {
    static_assert(!std::is_copy_constructible<reply_cache>::value && !std::is_copy_constructible<retained_reply>::value,
            "mutable epoch caches and payload owners cannot be copied");
    for (int field = 0; field < 7; ++field) {
        auto capacity = pageable();
        if (field == 0) { capacity.payload_bytes = 0; }
        if (field == 1) { capacity.payload_bytes = (512ull << 20) + 1; }
        if (field == 2) { capacity.entries = 0; }
        if (field == 3) { capacity.entries = 17; }
        if (field == 4) { capacity.page_bytes = 1; }
        if (field == 5) { capacity.memory = reply_memory::locked; }
        if (field == 6) { capacity.memory = static_cast<reply_memory>(99); }
        refused([&] { reply_cache invalid(0, stamp(), bounds(), capacity); });
    }
    reply_cache cache(0, stamp(), bounds(), pageable());
    step_fixture fixture;
    refused([&] { cache.retain_applied(fixture.certificate.owners[1], fixture.payload()); });
    refused([&] { cache.retain_applied(fixture.certificate.owners[0], {nullptr,64}); });
    refused([&] { cache.retain_applied(fixture.certificate.owners[0], {fixture.payload().data,0}); });
    refused([&] { cache.retain_applied(fixture.certificate.owners[0], {fixture.payload().data,size_t(max_chunk_bytes)+1}); });
    auto stale = fixture.certificate.owners[0]; ++stale.step.stamp.epoch;
    refused([&] { cache.retain_applied(stale, fixture.payload()); });
    require(!cache.stats().entries, "invalid input allocated a reply");
    auto skipped = fixture.certificate.owners[0]; ++skipped.step.transaction;
    refused([&] { cache.retain_applied(skipped, fixture.payload()); });
    require(!cache.stats().entries, "skipped transaction allocated a reply");
}

static void native_locks() {
    glm_fixture_lock_quota quota;
    const auto before = glm_stage::locked_pages::locked_stats();
    step_fixture fixture(1,4097);
    const auto pages = glm_stage::locked_pages::required_bytes(fixture.payload().size);
    reply_lease retained;
    {
        reply_cache cache(0, stamp(), bounds(), {8194,2*pages,2,reply_memory::locked});
        retained = cache.retain_applied(fixture.certificate.owners[0], fixture.payload());
        cache.commit(fixture.certificate); retained->verify_locks();
        require(retained->pages_locked() && cache.stats().page_bytes == pages && cache.stats().payload_bytes == 4097,
                "locked reply lost native page rounding or payload accounting");
    }
    retained->verify_locks();
    require(hash(retained->payload()) == fixture.certificate.owners[0].output_digest &&
            glm_stage::locked_pages::locked_stats().bytes == before.bytes + pages, "cache destruction freed a leased native reply");
    retained.reset();
    const auto after = glm_stage::locked_pages::locked_stats();
    require(before.bytes == after.bytes && before.buffers == after.buffers, "native reply leaked lock ownership");
    reply_cache too_small(0, stamp(), bounds(), {8194,pages-1,2,reply_memory::locked});
    refused([&] { too_small.retain_applied(fixture.certificate.owners[0], fixture.payload()); });
    require(!too_small.stats().entries && !too_small.stats().page_bytes && too_small.stats().refusals == 1,
            "page-budget refusal fell back to pageable memory or leaked charge");
}

static void conflicts_and_revocation() {
    for (int field = 0; field < 3; ++field) {
        reply_cache cache(0, stamp(), bounds(), pageable());
        step_fixture fixture;
        auto lease = cache.retain_applied(fixture.certificate.owners[0], fixture.payload()); cache.commit(fixture.certificate);
        if (field == 0) { cache.invalidate(); }
        if (field == 1) {
            fixture.certificate.owners[2].output_digest[0] ^= 1;
            refused([&] { cache.commit(fixture.certificate); });
        }
        if (field == 2) {
            fixture.outputs[0][0] ^= 1;
            refused([&] { cache.retain_applied(fixture.certificate.owners[0], fixture.payload()); });
        }
        refused([&] { lease->payload(); }); refused([&] { lease->receipt(); });
        refused([&] { cache.committed_reply(fixture.certificate.owners[0].step); });
        require(cache.stats().payload_bytes == 64, "revoked external lease became unaccounted");
        lease.reset(); require(!cache.stats().payload_bytes, "revoked final lease did not release");
    }
}

static void concurrent_retention() {
    reply_cache cache(0, stamp(), bounds(), pageable(128,2));
    step_fixture first, second(2);
    auto initial = cache.retain_applied(first.certificate.owners[0], first.payload()); cache.commit(first.certificate);
    std::weak_ptr<const retained_reply> weak = initial;
    initial.reset();
    std::atomic<bool> go{false}, stop{false}, failed{false};
    std::atomic<unsigned> acquired{0}, participating{0};
    std::vector<std::thread> readers;
    for (unsigned i = 0; i < 8; ++i) {
        readers.emplace_back([&,i] {
            while (!go.load()) { std::this_thread::yield(); }
            while (!stop.load()) {
                if (auto lease = weak.lock()) {
                    ++acquired;
                    participating.fetch_or(1u << i);
                    if (hash(lease->payload()) != first.certificate.owners[0].output_digest) { failed = true; }
                }
            }
        });
    }
    go = true;
    while (participating.load() != 255) { std::this_thread::yield(); }
    cache.retain_applied(second.certificate.owners[0], second.payload()); cache.commit(second.certificate);
    for (unsigned i = 0; i < 1000; ++i) {
        const auto stats = cache.stats();
        if (stats.payload_bytes > 128 || stats.entries > 2) { failed = true; }
    }
    stop = true;
    for (auto & reader : readers) { reader.join(); }
    require(!failed && acquired >= 8 && weak.expired() && cache.stats().payload_bytes == 64 && cache.stats().entries == 1,
            "concurrent weak retention escaped reservation/final-release accounting");
}

int main() {
    try {
        lifecycle(); capacities(); malformed(); native_locks(); conflicts_and_revocation(); concurrent_retention();
        std::puts("GLM replies: six suites passed (immutable copies, retry/expiry, shared caps, real native locks, revocation/certificates, concurrent retired leases); full residency NOT_VALIDATED");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM reply test failed: %s\n", error.what()); return 1;
    }
}
