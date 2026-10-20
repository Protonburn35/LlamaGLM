#pragma once

#include "gguf.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace glm_stage {

constexpr uint32_t plan_version = 1;
constexpr uint32_t max_context = 1048576;

uint64_t checked_add(uint64_t a, uint64_t b);
uint64_t checked_mul(uint64_t a, uint64_t b);

struct layer_range {
    uint32_t begin;
    uint32_t end;
};

struct plan {
    std::array<layer_range, 3> stages;
    uint32_t context = max_context;

    static plan from_counts(const std::string & counts, uint32_t context = max_context);
    void validate(uint32_t layers) const;
    uint32_t owner(uint32_t layer) const;
};

struct geometry {
    uint32_t layers;
    uint32_t nextn_layers;
    uint32_t hidden;
    uint32_t streams;
    uint32_t heads;
    uint32_t kda_head;
    uint32_t conv_kernel;
    uint32_t kv_latent;
    uint32_t index_head;
    uint32_t kpool;
    uint32_t experts;
    uint32_t selected_experts;
    uint32_t context_limit;
    std::vector<bool> recurrent;

    static geometry from_gguf(const gguf_context * ctx);
    void validate() const;
    uint64_t boundary_bytes(uint32_t rows) const;
};

enum class tensor_role { trunk, input, output, draft };

struct tensor_owner {
    uint32_t rank;
    int32_t layer;
    tensor_role role;
    bool routed_experts;
};

tensor_owner assign_tensor(const plan & split, const geometry & model, const std::string & name);

struct state_budget {
    uint32_t kda_layers = 0;
    uint32_t dsa_layers = 0;
    uint64_t mla_bytes = 0;
    uint64_t index_bytes = 0;
    uint64_t recurrent_bytes = 0;

    uint64_t total() const;
};

// KDA state stays FP32, as in llama_model::create_memory; MLA/index rows use two-byte storage.
std::array<state_budget, 3> state_budgets(const plan & split, const geometry & model, uint32_t rollback_rows = 0);

} // namespace glm_stage
