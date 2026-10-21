#pragma once

#include "stage-session-codec.h"

#include <memory>

namespace glm_rpc {

enum class sampling_mode : uint32_t { greedy = 1, temperature_distribution = 2 };

struct sampling_config {
    sampling_mode mode = sampling_mode::greedy;
    uint32_t vocabulary = 0;
    uint32_t seed = 0;
    float temperature = 1;
    uint32_t candidates = 8;
};

// Canonical versioned identity of every supported sampling parameter. Random sentinel
// seeds, unknown policies and implicit configuration changes refuse.
using sampling_config_wire = std::array<uint8_t,32>;
sampling_config_wire canonical_sampling(const sampling_config & config);
digest fingerprint_sampling(const sampling_config & config);

struct sampling_result {
    owner_receipt receipt;
    digest configuration{}, certificate_digest{};
    uint64_t ordinal = 0;
    uint32_t selected_token = 0, count = 0, row = 0;
    float selected_logit = 0;
    std::array<terminal_candidate,8> top{};
};

// Single-writer C-side native CPU sampler. Draws use a clone; committed RNG changes
// only on complete_after_global. Caller must prove global completion separately.
// Returned pending facts are not publishable. Wire v1 still permits greedy only.
class sampling_transactions {
public:
    sampling_transactions(const sampling_config & config, const epoch_stamp & stamp, const limits & bounds);
    ~sampling_transactions();
    sampling_transactions(const sampling_transactions &) = delete;
    sampling_transactions & operator=(const sampling_transactions &) = delete;
    // Exact canonical LE FP32 logits for all rows, bound to the actual C receipt.
    sampling_result draw(const owner_receipt & terminal, bytes_view logits);
    sampling_result complete_after_global(const commit_certificate & certificate);
    void invalidate() noexcept;
    uint64_t committed_draws() const;
private:
    struct impl;
    std::unique_ptr<impl> state;
};

} // namespace glm_rpc
