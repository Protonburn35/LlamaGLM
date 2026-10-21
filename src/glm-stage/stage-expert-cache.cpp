#include "stage-storage.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <iterator>
#include <list>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace glm_stage {
namespace {

struct payload_ledger {
    std::atomic<uint64_t> bytes{0};
    std::atomic<size_t> entries{0};
};

struct payload_delete {
    std::shared_ptr<payload_ledger> ledger;
    uint64_t bytes;
    void operator()(std::vector<uint8_t> * data) const {
        delete data;
        ledger->bytes -= bytes;
        --ledger->entries;
    }
};

struct reservation {
    std::shared_ptr<payload_ledger> ledger;
    uint64_t bytes;
    bool pending = true;
    ~reservation() { if (pending) { ledger->bytes -= bytes; --ledger->entries; } }
};

} // namespace

struct desktop_experts::impl {
    struct entry {
        bool loading = true;
        expert_lease payload;
        std::condition_variable changed;
        std::list<expert_address>::iterator recent;
    };
    std::vector<std::shared_ptr<expert_file>> files;
    std::set<expert_address> catalog;
    uint64_t budget;
    size_t entry_budget, max_active;
    mutable std::mutex mutex;
    std::map<expert_address, std::shared_ptr<entry>> entries;
    std::list<expert_address> recent;
    std::shared_ptr<payload_ledger> ledger = std::make_shared<payload_ledger>();
    bool stopped = false;
    std::exception_ptr failure;
    size_t active = 0, waiters = 0;
    expert_cache_stats counters{};

    impl(uint32_t rank, std::vector<std::shared_ptr<expert_file>> sources, const std::vector<expert_address> & owned,
         uint64_t byte_budget, size_t entry_budget, size_t max_active_reads) :
        files(std::move(sources)), budget(byte_budget), entry_budget(entry_budget), max_active(max_active_reads) {
        if ((rank != 0 && rank != 2) || files.empty() || files.size() > 100 || owned.empty() || owned.size() > 100000 ||
                budget == 0 || budget > (1ull << 40) || entry_budget == 0 || entry_budget > 4096 || max_active == 0 || max_active > 8) {
            throw std::invalid_argument("invalid desktop expert cache policy/budgets; laptop disk fallback is forbidden");
        }
        for (const auto & file : files) { if (!file) { throw std::invalid_argument("desktop expert source is absent"); } }
        for (const auto & address : owned) {
            if (address.file >= files.size() || address.bytes == 0 || address.bytes > max_expert_read_bytes || address.bytes > budget ||
                    address.offset > files[address.file]->size() || address.bytes > files[address.file]->size() - address.offset ||
                    !catalog.insert(address).second) { throw std::invalid_argument("invalid or duplicate owned desktop expert span"); }
        }
        uint32_t file = 0;
        uint64_t end = 0;
        bool first = true;
        for (const auto & address : catalog) {
            if (!first && address.file == file && address.offset < end) { throw std::invalid_argument("overlapping desktop expert spans"); }
            first = false; file = address.file; end = address.offset + address.bytes;
        }
    }

    void healthy() const {
        if (stopped) { throw std::runtime_error("desktop expert cache was stopped"); }
        if (failure) { std::rethrow_exception(failure); }
    }

    [[noreturn]] void refuse(const char * message) {
        ++counters.refusals;
        throw std::runtime_error(message);
    }

    void make_room(uint64_t bytes) {
        while (ledger->bytes.load() > budget - bytes || ledger->entries.load() >= entry_budget) {
            bool removed = false;
            for (auto it = recent.begin(); it != recent.end(); ++it) {
                const auto found = entries.find(*it);
                if (found->second.use_count() == 1 && found->second->payload.use_count() == 1) {
                    recent.erase(it);
                    entries.erase(found);
                    ++counters.evictions;
                    removed = true;
                    break;
                }
            }
            // Retained leases, including a concurrent weak-lock winner, remain charged until deletion.
            if (!removed) { refuse("desktop expert cache is full with active leases/reads"); }
        }
    }

    expert_lease acquire(expert_address address) {
        std::shared_ptr<entry> value;
        {
            std::unique_lock<std::mutex> lock(mutex);
            healthy();
            if (!catalog.count(address)) { refuse("selected expert is outside the complete owned catalog"); }
            const auto found = entries.find(address);
            if (found != entries.end()) {
                value = found->second;
                if (value->loading) {
                    if (waiters >= 32) { refuse("desktop expert waiters exceed local limit"); }
                    ++waiters; ++counters.deduplicated;
                    try { value->changed.wait(lock, [&] { return !value->loading || stopped || failure; }); }
                    catch (...) { --waiters; throw; }
                    --waiters;
                    healthy();
                }
                recent.splice(recent.end(), recent, value->recent);
                ++counters.hits;
                return value->payload;
            }
            if (active >= max_active) { refuse("desktop expert native reads exceed local limit"); }
            make_room(address.bytes);
            value = std::make_shared<entry>();
            entries.emplace(address, value);
            const uint64_t allocated = (ledger->bytes += address.bytes);
            ++ledger->entries;
            ++active; ++counters.misses;
            counters.peak_reserved_payload_bytes = std::max(counters.peak_reserved_payload_bytes, allocated);
        }

        try {
            reservation reserved{ledger, address.bytes};
            std::unique_ptr<std::vector<uint8_t>, payload_delete> data(new std::vector<uint8_t>(size_t(address.bytes)),
                payload_delete{ledger, address.bytes});
            reserved.pending = false;
            files[address.file]->read_exact(address.offset, data->data(), data->size());
            expert_lease payload(std::move(data));
            std::lock_guard<std::mutex> lock(mutex);
            healthy();
            recent.push_back(address);
            value->recent = std::prev(recent.end());
            value->payload = std::move(payload);
            value->loading = false;
            --active;
            value->changed.notify_all();
            return value->payload;
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            failure = std::current_exception();
            ++counters.failures;
            value->loading = false;
            --active;
            for (const auto & item : entries) { item.second->changed.notify_all(); }
            throw;
        }
    }
};

desktop_experts::desktop_experts(uint32_t rank, std::vector<std::shared_ptr<expert_file>> sources,
        const std::vector<expert_address> & owned, uint64_t byte_budget, size_t entry_budget, size_t max_active_reads) :
    state(std::make_unique<impl>(rank, std::move(sources), owned, byte_budget, entry_budget, max_active_reads)) {}

desktop_experts::~desktop_experts() = default;
expert_lease desktop_experts::acquire(expert_address address) { return state->acquire(address); }

void desktop_experts::stop() {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->stopped = true;
    for (const auto & item : state->entries) { item.second->changed.notify_all(); }
}

expert_cache_stats desktop_experts::stats() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto result = state->counters;
    result.reserved_payload_bytes = state->ledger->bytes.load();
    result.reserved_entries = state->ledger->entries.load();
    result.active_reads = state->active;
    return result;
}

} // namespace glm_stage
