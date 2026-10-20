#include "stage-plan.h"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void refuses(const std::function<void()> & operation) {
    try {
        operation();
    } catch (const std::invalid_argument &) {
        return;
    } catch (const std::overflow_error &) {
        return;
    }
    throw std::runtime_error("invalid input was accepted");
}

static glm_stage::geometry model_geometry() {
    glm_stage::geometry model{45, 1, 4096, 4, 64, 128, 4, 512, 128, 4, 288, 8, 1048576, {}};
    for (uint32_t i = 0; i < model.layers; ++i) {
        model.recurrent.push_back(i % 4 != 3);
    }
    return model;
}

int main() {
    using namespace glm_stage;
    try {
        const auto model = model_geometry();
        const auto split = plan::from_counts("22,5,18");
        split.validate(45);
        for (uint32_t layer = 0; layer < 45; ++layer) {
            require(split.owner(layer) == (layer < 22 ? 0U : layer < 27 ? 1U : 2U), "wrong layer owner");
        }
        for (const auto * bad : {"22,5", "22,5,18,1", "22,0,23", "-1,5,41", "22, 5,18", "22,5,18x", "4294967295,5,18"}) {
            refuses([&] { plan::from_counts(bad); });
        }
        refuses([&] { plan::from_counts("22,5,17").validate(45); });
        refuses([&] { plan::from_counts("22,5,18", 0); });
        refuses([&] { plan::from_counts("22,5,18", 1048577); });
        refuses([&] { split.owner(45); });
        auto broken = split;
        broken.stages[1].begin = 21;
        refuses([&] { broken.validate(45); });
        broken.stages[1].begin = 23;
        refuses([&] { broken.validate(45); });

        require(assign_tensor(split, model, "token_embd.weight").rank == 0, "embedding owner");
        require(assign_tensor(split, model, "output.weight").rank == 2, "output owner");
        require(assign_tensor(split, model, "blk.45.nextn.eh_proj.weight").role == tensor_role::draft, "draft owner");
        require(assign_tensor(split, model, "blk.23.ffn_up_exps.weight").routed_experts, "expert classification");
        require(!assign_tensor(split, model, "blk.23.ffn_gate_inp.weight").routed_experts, "router is not an expert tensor");
        for (const auto * bad : {"visual.weight", "blk.-1.attn.weight", "blk.46.attn.weight", "blk.023.ffn.weight", "blk.23", "blk.23."}) {
            refuses([&] { assign_tensor(split, model, bad); });
        }

        require(model.boundary_bytes(1) == 65536, "FP32 four-stream boundary");
        require(model.boundary_bytes(2048) == 134217728, "prefill boundary");
        refuses([&] { model.boundary_bytes(0); });
        refuses([&] { model.boundary_bytes(1048577); });
        for (const uint32_t context : {131072U, 262144U, 524288U, 1048576U}) {
            const auto state = state_budgets(plan::from_counts("22,5,18", context), model);
            require(state[0].dsa_layers == 5 && state[1].dsa_layers == 1 && state[2].dsa_layers == 5, "DSA ownership");
            require(state[0].kda_layers == 17 && state[1].kda_layers == 4 && state[2].kda_layers == 13, "KDA ownership");
            // Independent byte arithmetic: latent 512 FP16 values + key/gate/pooled 3 x 128 FP16 values.
            require(state[1].mla_bytes + state[1].index_bytes == uint64_t(context) * 1792, "sparse cache bytes");
            require(state[1].recurrent_bytes == 4ULL * (1048576 + 73728) * 4, "FP32 recurrent and conv bytes");
        }
        const auto state = state_budgets(split, model);
        const auto rollback = state_budgets(split, model, 3);
        require(rollback[1].recurrent_bytes == 4 * state[1].recurrent_bytes, "rollback snapshots");
        require(rollback[1].mla_bytes == state[1].mla_bytes, "rollback does not duplicate MLA cache");
        require(state[1].total() == state[1].mla_bytes + state[1].index_bytes + state[1].recurrent_bytes, "state total");
        auto bad_model = model;
        bad_model.selected_experts = 6;
        refuses([&] { state_budgets(split, bad_model); });
        bad_model = model;
        bad_model.recurrent[23] = true;
        refuses([&] { state_budgets(split, bad_model); });
        refuses([&] { checked_mul(std::numeric_limits<uint64_t>::max(), 2); });
        refuses([&] { checked_add(std::numeric_limits<uint64_t>::max(), 1); });

        gguf_context * metadata = gguf_init_empty();
        gguf_set_val_str(metadata, "general.architecture", "glm5-next");
        gguf_set_val_str(metadata, "glm5-next.block_count", "46");
        refuses([&] { geometry::from_gguf(metadata); });
        gguf_free(metadata);
        std::cout << "PASS: ownership, malformed plans, tensor roles, long-context state, precision and overflow\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
