#pragma once

#include "stage-session.h"

#include <memory>

namespace glm_rpc {

enum class reply_memory { pageable, locked };

struct reply_limits {
    uint64_t payload_bytes = 0;
    uint64_t page_bytes = 0;
    uint32_t entries = 0;
    reply_memory memory = reply_memory::pageable;
};

struct reply_stats {
    uint64_t payload_bytes, peak_payload_bytes, page_bytes, peak_page_bytes;
    uint32_t entries;
    uint64_t refusals, failures;
};

class retained_reply {
public:
    ~retained_reply();
    retained_reply(const retained_reply &) = delete;
    retained_reply & operator=(const retained_reply &) = delete;
    const owner_receipt & receipt() const;
    // Immutable unfragmented bytes. Explicit invalidation revokes future access, but cannot
    // retract borrowed pointers: stop/join consumers and discard old views before invalidating.
    bytes_view payload() const;
    bool pages_locked() const;
    void verify_locks() const;
private:
    friend class reply_cache;
    struct impl;
    explicit retained_reply(std::unique_ptr<impl> state);
    std::unique_ptr<impl> state;
};

using reply_lease = std::shared_ptr<const retained_reply>;

// Single-writer epoch cache: one applied output plus one previous locally committed reply.
// Capacity includes externally retained retired leases until final deletion. Leases may be
// read/released concurrently. No disk spill, re-execution, implicit epoch reset or heap fallback.
// Local commitment is not publication: the coordinator barrier still requires all owners.
class reply_cache {
public:
    reply_cache(uint32_t rank, const epoch_stamp & stamp, const limits & bounds, const reply_limits & capacity);
    ~reply_cache();
    reply_cache(const reply_cache &) = delete;
    reply_cache & operator=(const reply_cache &) = delete;
    reply_lease retain_applied(const owner_receipt & receipt, bytes_view payload);
    reply_lease applied_reply(const step_key & expected) const;
    reply_lease commit(const commit_certificate & certificate);
    reply_lease committed_reply(const step_key & expected) const;
    void invalidate() noexcept;
    reply_stats stats() const;
private:
    struct impl;
    std::unique_ptr<impl> state;
};

} // namespace glm_rpc
