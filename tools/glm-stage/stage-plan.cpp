#include "stage-plan.h"

#include <charconv>
#include <limits>
#include <stdexcept>

namespace glm_stage {

uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::overflow_error("stage byte count overflow");
    }
    return a + b;
}

uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::overflow_error("stage byte count overflow");
    }
    return a * b;
}

static uint32_t parse_u32(const std::string & text) {
    uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("expected an unsigned 32-bit integer");
    }
    return value;
}

plan plan::from_counts(const std::string & counts, uint32_t context) {
    plan result{};
    result.context = context;
    size_t start = 0;
    uint64_t offset = 0;
    for (size_t rank = 0; rank < result.stages.size(); ++rank) {
        const size_t comma = counts.find(',', start);
        if ((rank < 2) != (comma != std::string::npos)) {
            throw std::invalid_argument("split must contain exactly three layer counts");
        }
        const uint32_t count = parse_u32(counts.substr(start, comma == std::string::npos ? comma : comma - start));
        const uint64_t end = checked_add(offset, count);
        if (count == 0 || end > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("each stage must own a nonempty valid layer range");
        }
        result.stages[rank] = {uint32_t(offset), uint32_t(end)};
        offset = end;
        start = comma == std::string::npos ? counts.size() : comma + 1;
    }
    result.validate(uint32_t(offset));
    return result;
}

void plan::validate(uint32_t layers) const {
    if (context == 0 || context > max_context || layers < 3) {
        throw std::invalid_argument("invalid context or layer count");
    }
    uint32_t next = 0;
    for (const auto & range : stages) {
        if (range.begin != next || range.end <= range.begin || range.end > layers) {
            throw std::invalid_argument("three stages must cover every layer once, with no gaps or overlap");
        }
        next = range.end;
    }
    if (next != layers) {
        throw std::invalid_argument("three stages do not cover the complete model");
    }
}

uint32_t plan::owner(uint32_t layer) const {
    validate(stages.back().end);
    for (uint32_t rank = 0; rank < stages.size(); ++rank) {
        if (layer >= stages[rank].begin && layer < stages[rank].end) {
            return rank;
        }
    }
    throw std::invalid_argument("layer is outside the target graph");
}

static uint32_t read_u32(const gguf_context * ctx, const char * suffix) {
    const std::string key = std::string("glm5-next.") + suffix;
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        throw std::invalid_argument("missing or incorrectly typed metadata: " + key);
    }
    return gguf_get_val_u32(ctx, id);
}

geometry geometry::from_gguf(const gguf_context * ctx) {
    const int64_t arch = gguf_find_key(ctx, "general.architecture");
    if (arch < 0 || gguf_get_kv_type(ctx, arch) != GGUF_TYPE_STRING ||
            std::string(gguf_get_val_str(ctx, arch)) != "glm5-next") {
        throw std::invalid_argument("expected a glm5-next GGUF");
    }
    geometry result{};
    const uint32_t all_layers = read_u32(ctx, "block_count");
    result.nextn_layers     = read_u32(ctx, "nextn_predict_layers");
    if (all_layers <= result.nextn_layers || all_layers > 256) {
        throw std::invalid_argument("invalid trunk/NextN layer counts");
    }
    result.layers           = all_layers - result.nextn_layers;
    result.hidden           = read_u32(ctx, "embedding_length");
    result.streams          = read_u32(ctx, "hyper_connection.count");
    result.heads            = read_u32(ctx, "attention.head_count");
    result.kda_head         = read_u32(ctx, "kda.head_dim");
    result.conv_kernel      = read_u32(ctx, "ssm.conv_kernel");
    result.kv_latent        = read_u32(ctx, "attention.kv_lora_rank");
    result.index_head       = read_u32(ctx, "attention.indexer.key_length");
    result.kpool            = read_u32(ctx, "attention.indexer.kpool");
    result.experts          = read_u32(ctx, "expert_count");
    result.selected_experts = read_u32(ctx, "expert_used_count");
    result.context_limit    = read_u32(ctx, "context_length");

    const int64_t kv = gguf_find_key(ctx, "glm5-next.attention.head_count_kv");
    if (kv < 0 || gguf_get_kv_type(ctx, kv) != GGUF_TYPE_ARRAY ||
            (gguf_get_arr_type(ctx, kv) != GGUF_TYPE_UINT32 && gguf_get_arr_type(ctx, kv) != GGUF_TYPE_INT32) ||
            gguf_get_arr_n(ctx, kv) != all_layers) {
        throw std::invalid_argument("expected one 32-bit integer KV head count per trunk/NextN layer");
    }
    const auto head_at = [&](uint32_t i) -> int64_t {
        const void * data = gguf_get_arr_data(ctx, kv);
        return gguf_get_arr_type(ctx, kv) == GGUF_TYPE_INT32 ?
            int64_t(static_cast<const int32_t *>(data)[i]) : int64_t(static_cast<const uint32_t *>(data)[i]);
    };
    for (uint32_t i = 0; i < result.layers; ++i) {
        if (head_at(i) != 0 && head_at(i) != 1) {
            throw std::invalid_argument("expected recurrent layers or single-head latent MLA");
        }
        result.recurrent.push_back(head_at(i) == 0);
    }
    for (uint32_t i = result.layers; i < all_layers; ++i) {
        if (head_at(i) != 1) {
            throw std::invalid_argument("expected DSA NextN layers");
        }
    }
    result.validate();
    return result;
}

void geometry::validate() const {
    if (layers != 45 || nextn_layers != 1 || hidden != 4096 || streams != 4 || heads != 64 ||
            kda_head != 128 || conv_kernel != 4 || kv_latent != 512 || index_head != 128 || kpool != 4 ||
            experts != 288 || selected_experts != 8 || context_limit != max_context || recurrent.size() != layers) {
        throw std::invalid_argument("GGUF geometry differs from the audited GLM-5.3-Flash layout");
    }
    for (uint32_t i = 0; i < layers; ++i) {
        if (recurrent[i] != (i % 4 != 3)) {
            throw std::invalid_argument("GGUF attention schedule differs from the audited GLM layout");
        }
    }
}

uint64_t geometry::boundary_bytes(uint32_t rows) const {
    validate();
    if (rows == 0 || rows > context_limit) {
        throw std::invalid_argument("invalid boundary row count");
    }
    return checked_mul(checked_mul(hidden, streams), checked_mul(rows, sizeof(float)));
}

tensor_owner assign_tensor(const plan & split, const geometry & model, const std::string & name) {
    model.validate();
    split.validate(model.layers);
    if (name == "token_embd.weight") {
        return {0, -1, tensor_role::input, false};
    }
    if (name == "output.weight" || name == "output_norm.weight") {
        return {2, -1, tensor_role::output, false};
    }
    if (name.compare(0, 4, "blk.") != 0) {
        throw std::invalid_argument("unclassified tensor: " + name);
    }
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos || dot + 1 == name.size()) {
        throw std::invalid_argument("malformed layer tensor name: " + name);
    }
    const std::string digits = name.substr(4, dot - 4);
    const uint32_t layer = parse_u32(digits);
    if (std::to_string(layer) != digits) {
        throw std::invalid_argument("noncanonical layer tensor name: " + name);
    }
    const std::string suffix = name.substr(dot + 1);
    const bool expert = suffix == "ffn_gate_exps.weight" || suffix == "ffn_down_exps.weight" ||
                        suffix == "ffn_up_exps.weight";
    if (layer < model.layers) {
        return {split.owner(layer), int32_t(layer), tensor_role::trunk, expert};
    }
    if (layer - model.layers < model.nextn_layers) {
        return {2, int32_t(layer), tensor_role::draft, expert};
    }
    throw std::invalid_argument("tensor layer is outside the model: " + name);
}

uint64_t state_budget::total() const {
    return checked_add(checked_add(mla_bytes, index_bytes), recurrent_bytes);
}

expert_span expert_file_span(const geometry & model, const int64_t * shape, uint64_t tensor_offset,
                             uint64_t tensor_bytes, uint32_t expert) {
    model.validate();
    if (shape == nullptr || shape[2] != model.experts || shape[3] != 1 || expert >= model.experts ||
            tensor_bytes == 0 || tensor_bytes % model.experts != 0) {
        throw std::invalid_argument("invalid routed-expert tensor or expert index");
    }
    if (!((shape[0] == model.hidden && shape[1] == 2048) ||
            (shape[0] == 2048 && shape[1] == model.hidden))) {
        throw std::invalid_argument("unexpected routed-expert projection shape");
    }
    checked_add(tensor_offset, tensor_bytes);
    const uint64_t slab = tensor_bytes / model.experts;
    return {checked_add(tensor_offset, checked_mul(slab, expert)), slab};
}

std::array<state_budget, 3> state_budgets(const plan & split, const geometry & model, uint32_t rollback_rows) {
    model.validate();
    split.validate(model.layers);
    if (split.context > model.context_limit || rollback_rows > split.context) {
        throw std::invalid_argument("state capacity exceeds model context");
    }
    const uint64_t mla = checked_mul(checked_mul(model.kv_latent, split.context), 2);
    // The current indexer caches key, gate and pooled key in every token row.
    const uint64_t index = checked_mul(checked_mul(checked_mul(model.index_head, 3), split.context), 2);
    const uint64_t conv = checked_mul(checked_mul(3, model.conv_kernel - 1), checked_mul(model.heads, model.kda_head));
    const uint64_t recurrent = checked_mul(checked_mul(model.heads, model.kda_head), model.kda_head);
    const uint64_t state = checked_mul(checked_mul(checked_add(conv, recurrent), 4), checked_add(1, rollback_rows));
    std::array<state_budget, 3> result{};
    for (uint32_t layer = 0; layer < model.layers; ++layer) {
        auto & budget = result[split.owner(layer)];
        if (model.recurrent[layer]) {
            ++budget.kda_layers;
            budget.recurrent_bytes = checked_add(budget.recurrent_bytes, state);
        } else {
            ++budget.dsa_layers;
            budget.mla_bytes = checked_add(budget.mla_bytes, mla);
            budget.index_bytes = checked_add(budget.index_bytes, index);
        }
    }
    return result;
}

} // namespace glm_stage
