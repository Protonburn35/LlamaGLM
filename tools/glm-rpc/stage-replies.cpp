#include "stage-replies.h"
#include "stage-session-codec.h"
#include "stage-locked-pages.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace glm_rpc {
namespace {

constexpr uint64_t max_cache_bytes = 512ull << 20;

struct accounting {
    reply_limits capacity;
    mutable std::mutex mutex;
    reply_stats used{};
    std::atomic<bool> invalid{false};
    explicit accounting(const reply_limits & capacity) : capacity(capacity) {}
};

struct reservation {
    std::shared_ptr<accounting> account;
    uint64_t bytes, pages;
    reservation(std::shared_ptr<accounting> account, size_t bytes) : account(std::move(account)), bytes(bytes),
            pages(this->account->capacity.memory == reply_memory::locked ? glm_stage::locked_pages::required_bytes(bytes) : 0) {
        std::lock_guard<std::mutex> lock(this->account->mutex);
        auto & used = this->account->used;
        const auto & capacity = this->account->capacity;
        if (this->account->invalid || bytes > capacity.payload_bytes - used.payload_bytes ||
                pages > capacity.page_bytes - used.page_bytes || used.entries >= capacity.entries) {
            ++used.refusals;
            throw std::length_error("reply capacity exhausted, including retained leases");
        }
        used.payload_bytes += bytes; used.page_bytes += pages; ++used.entries;
        if (used.payload_bytes > used.peak_payload_bytes) { used.peak_payload_bytes = used.payload_bytes; }
        if (used.page_bytes > used.peak_page_bytes) { used.peak_page_bytes = used.page_bytes; }
    }
    reservation(const reservation &) = delete;
    reservation & operator=(const reservation &) = delete;
    ~reservation() {
        std::lock_guard<std::mutex> lock(account->mutex);
        account->used.payload_bytes -= bytes; account->used.page_bytes -= pages; --account->used.entries;
    }
};

bool same_stamp(const epoch_stamp & a, const epoch_stamp & b) {
    return a.identity == b.identity && a.session == b.session && a.epoch == b.epoch && a.generation == b.generation;
}

} // namespace

struct retained_reply::impl {
    // Reservation is destroyed last, after native/heap payload release.
    reservation charge;
    owner_receipt receipt;
    std::unique_ptr<uint8_t[]> heap;
    std::unique_ptr<glm_stage::locked_pages> locked;
    impl(std::shared_ptr<accounting> account, const owner_receipt & receipt, bytes_view payload) :
            charge(std::move(account), payload.size), receipt(receipt) {
        try {
            if (charge.pages) {
                locked = std::make_unique<glm_stage::locked_pages>(payload.size, charge.pages);
                std::memcpy(locked->data(), payload.data, payload.size); locked->verify_locked();
            } else {
                heap = std::make_unique<uint8_t[]>(payload.size);
                std::memcpy(heap.get(), payload.data, payload.size);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(charge.account->mutex);
            ++charge.account->used.failures;
            throw;
        }
    }
    void check() const {
        if (charge.account->invalid) { throw std::runtime_error("reply epoch is permanently invalid"); }
    }
};

retained_reply::retained_reply(std::unique_ptr<impl> state) : state(std::move(state)) {}
retained_reply::~retained_reply() = default;
const owner_receipt & retained_reply::receipt() const { state->check(); return state->receipt; }
bytes_view retained_reply::payload() const {
    state->check();
    return {state->locked ? state->locked->data() : state->heap.get(), size_t(state->charge.bytes)};
}
bool retained_reply::pages_locked() const { state->check(); return bool(state->locked); }
void retained_reply::verify_locks() const {
    state->check();
    if (!state->locked) { throw std::runtime_error("reply payload is explicitly pageable"); }
    state->locked->verify_locked();
}

struct reply_cache::impl {
    uint32_t rank;
    epoch_stamp stamp;
    limits bounds;
    std::shared_ptr<accounting> account;
    reply_lease pending, last;
    digest last_certificate{};
    impl(uint32_t rank, const epoch_stamp & stamp, const limits & bounds, const reply_limits & capacity) :
            rank(rank), stamp(stamp), bounds(bounds), account(std::make_shared<accounting>(capacity)) {
        validate_step({stamp,1,1,0,1,stamp.identity}, bounds);
        if (rank > 2 || !capacity.payload_bytes || capacity.payload_bytes > max_cache_bytes ||
                !capacity.entries || capacity.entries > 16 ||
                (capacity.memory != reply_memory::pageable && capacity.memory != reply_memory::locked) ||
                (capacity.memory == reply_memory::pageable && capacity.page_bytes) ||
                (capacity.memory == reply_memory::locked && (!capacity.page_bytes ||
                    capacity.page_bytes > glm_stage::locked_pages::required_bytes(size_t(max_cache_bytes))))) {
            throw std::invalid_argument("invalid bounded reply cache policy");
        }
    }
    void check(const step_key & key) const {
        if (account->invalid) { throw std::runtime_error("reply cache is permanently invalid"); }
        validate_step(key, bounds);
        if (!same_stamp(key.stamp, stamp)) { throw std::invalid_argument("reply belongs to another immutable epoch"); }
    }
    void invalidate() noexcept { account->invalid = true; pending.reset(); last.reset(); last_certificate = {}; }
    [[noreturn]] void conflict() { invalidate(); throw std::runtime_error("conflicting reply invalidated the epoch cache"); }
};

reply_cache::reply_cache(uint32_t rank, const epoch_stamp & stamp, const limits & bounds, const reply_limits & capacity) :
        state(std::make_unique<impl>(rank, stamp, bounds, capacity)) {}
reply_cache::~reply_cache() = default;

reply_lease reply_cache::retain_applied(const owner_receipt & receipt, bytes_view payload) {
    state->check(receipt.step); validate_receipt(receipt, state->bounds);
    if (receipt.rank != state->rank || !payload.data || !payload.size || payload.size > max_chunk_bytes) {
        throw std::invalid_argument("invalid reply owner or bounded payload");
    }
    if (sha256({payload}) != receipt.output_digest) { state->conflict(); }
    for (const auto & cached : {state->pending, state->last}) {
        if (cached && cached->receipt().step.transaction == receipt.step.transaction) {
            if (!(cached->receipt() == receipt) || cached->payload().size != payload.size) { state->conflict(); }
            return cached;
        }
    }
    const auto previous = state->last ? state->last->receipt() : owner_receipt{};
    if (state->pending || receipt.step.transaction != previous.step.transaction + 1 ||
            receipt.step.chunk != previous.step.chunk + 1 || receipt.step.position != previous.end_position) {
        throw std::invalid_argument("reply busy, skipped, expired or out of position");
    }
    auto allocation = std::make_unique<retained_reply::impl>(state->account, receipt, payload);
    const auto copied = allocation->locked ? allocation->locked->data() : allocation->heap.get();
    if (sha256({{copied, payload.size}}) != receipt.output_digest) { state->conflict(); }
    state->pending = reply_lease(new retained_reply(std::move(allocation)));
    return state->pending;
}

reply_lease reply_cache::applied_reply(const step_key & expected) const {
    state->check(expected);
    if (!state->pending || !(state->pending->receipt().step == expected)) { throw std::invalid_argument("no matching applied reply"); }
    return state->pending;
}

reply_lease reply_cache::commit(const commit_certificate & certificate) {
    const auto & receipt = certificate.owners[state->rank]; state->check(receipt.step);
    if (state->last && receipt.step.transaction < state->last->receipt().step.transaction) {
        throw std::invalid_argument("committed reply certificate has expired");
    }
    if ((!state->last || receipt.step.transaction != state->last->receipt().step.transaction) &&
            (!state->pending || receipt.step.transaction != state->pending->receipt().step.transaction)) {
        throw std::invalid_argument("commit requires a retained applied reply");
    }
    digest fingerprint{};
    try { fingerprint = fingerprint_commit(certificate, state->bounds); } catch (...) { state->invalidate(); throw; }
    if (state->last && state->last->receipt().step.transaction == receipt.step.transaction) {
        if (!(state->last->receipt() == receipt) || fingerprint != state->last_certificate) { state->conflict(); }
        return state->last;
    }
    if (!state->pending || !(state->pending->receipt() == receipt)) { state->conflict(); }
    state->last = std::move(state->pending);
    state->last_certificate = fingerprint;
    return state->last;
}

reply_lease reply_cache::committed_reply(const step_key & expected) const {
    state->check(expected);
    if (!state->last || !(state->last->receipt().step == expected)) { throw std::invalid_argument("committed reply expired or differs"); }
    return state->last;
}

void reply_cache::invalidate() noexcept { state->invalidate(); }
reply_stats reply_cache::stats() const { std::lock_guard<std::mutex> lock(state->account->mutex); return state->account->used; }

} // namespace glm_rpc
