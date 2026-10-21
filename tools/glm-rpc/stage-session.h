#pragma once

#include "stage-wire.h"

#include <optional>

namespace glm_rpc {

struct epoch_stamp {
    digest identity{};
    session_id session{};
    uint64_t epoch = 0;
    uint64_t generation = 0;
};

struct step_key {
    epoch_stamp stamp;
    uint64_t transaction = 0;
    uint64_t chunk = 0;
    uint64_t position = 0;
    uint32_t rows = 0;
    digest request_digest{};
};

struct owner_receipt {
    step_key step;
    uint32_t rank = 0;
    uint64_t end_position = 0;
    digest input_digest{};
    digest output_digest{};
};

struct commit_certificate {
    std::array<owner_receipt, 3> owners{};
};

struct commit_acknowledgement {
    owner_receipt owner;
    digest certificate_digest{};
};

bool operator==(const step_key & a, const step_key & b);
bool operator==(const owner_receipt & a, const owner_receipt & b);

enum class owner_phase { empty, idle, running, applied, invalid };
enum class step_disposition { execute, running, applied, committed };

// Single-threaded local ledger, independent of connection frame counters. These are checked
// local facts, not a peer decoder, payload cache or proof that a graph actually executed.
// Callers authenticate rank/identity, hash actual inputs/outputs, retain bounded retry replies
// and stop/invalidate/recreate real contexts before fresh_context. Global commit needs all
// three commit acknowledgements; any uncertain outcome requires all-owner reset/replay.
class owner_transactions {
public:
    owner_transactions(uint32_t rank, const digest & identity, const limits & bounds);
    owner_transactions(const owner_transactions &) = delete;
    owner_transactions & operator=(const owner_transactions &) = delete;

    void fresh_context(const epoch_stamp & stamp);
    step_disposition prepare(const step_key & step, const digest & input_digest);
    owner_receipt applied(const step_key & step, const digest & output_digest);
    owner_receipt commit(const commit_certificate & certificate);
    void invalidate() noexcept;

    owner_phase phase() const { return state; }
    uint64_t committed_position() const { return position; }

private:
    uint32_t rank;
    digest identity;
    limits bounds;
    epoch_stamp current;
    owner_phase state = owner_phase::empty;
    uint64_t position = 0, next_transaction = 1, next_chunk = 1;
    std::optional<owner_receipt> pending;
    std::optional<commit_certificate> last;

    void check_epoch(const epoch_stamp & stamp) const;
    void check_step(const step_key & step) const;
    [[noreturn]] void conflict();
};

// Checks exact A/B/C ownership, common coordinates, complete state positions and the
// A output -> B input -> C input digest chain. Does not authenticate the receipts.
void validate_commit(const commit_certificate & certificate, const limits & bounds);
// Validates one applied receipt; adjacency and all-owner agreement require the certificate.
void validate_receipt(const owner_receipt & receipt, const limits & bounds);
void validate_step(const step_key & step, const limits & bounds);

} // namespace glm_rpc
