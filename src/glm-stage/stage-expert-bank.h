#pragma once

#include "stage-storage.h"
#include "ggml-backend.h"

#include <functional>

namespace glm_stage {

struct expert_projection {
    ggml_type type;
    int64_t input;
    int64_t output;
    std::vector<expert_address> experts;
};

struct expert_bank_stats {
    uint64_t materializations = 0;
    uint64_t slot_hits = 0;       // distinct complete expert groups reused
    uint64_t slot_misses = 0;     // distinct complete expert groups filled
    uint64_t copied_slabs = 0;    // completed per-projection copies, including failed prefixes
    uint64_t copied_bytes = 0;
    uint64_t failures = 0;
};

// Three raw gate/up/down banks with bounded slots. Backend must outlive the bank and all leases.
// Source/catalog bytes must be immutable. This does not establish physical residency or admission.
class expert_bank {
    struct impl;
public:
    class lease {
    public:
        lease(lease &&) noexcept = default;
        lease & operator=(lease &&) = delete;
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
        ~lease(); // completes backend work before releasing the bank for replacement
        ggml_tensor * projection(size_t index) const;
        const std::vector<int32_t> & remapped_ids() const;
    private:
        friend class expert_bank;
        lease(std::shared_ptr<impl> state, std::vector<int32_t> ids) noexcept;
        std::shared_ptr<impl> state;
        std::vector<int32_t> ids;
    };

    expert_bank(ggml_backend_t backend, uint32_t slots, const std::vector<expert_projection> & projections,
                uint64_t bank_budget, std::function<expert_lease(expert_address)> source);
    expert_bank(const expert_bank &) = delete;
    expert_bank & operator=(const expert_bank &) = delete;
    uint64_t buffer_bytes() const;
    expert_bank_stats stats() const;
    static bool supports_type(ggml_type type);
    // Metadata for graph construction; execution must depend on materialization and retain its lease.
    ggml_tensor * projection_for_graph(size_t index) const;
    // Preserves selected order/duplicates. Excess distinct IDs refuse; no routing clipping or waves.
    // Any materialization failure invalidates this bank. Busy banks refuse without overwriting leases.
    lease materialize(const std::vector<int32_t> & selected);

private:
    std::shared_ptr<impl> state;
};

} // namespace glm_stage
