#include "stage-expert-bank.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <array>
#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>

namespace glm_stage {

static bool bank_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_0: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q3_K: case GGML_TYPE_Q2_K: return true;
        default: return false;
    }
}

struct expert_bank::impl {
    ggml_backend_t backend;
    uint32_t slots;
    uint32_t experts;
    std::vector<expert_projection> specs;
    std::function<expert_lease(expert_address)> source;
    std::atomic<bool> in_use{false};
    bool failed = false;
    ggml_context_ptr context;
    std::array<ggml_tensor *, 3> banks{};
    ggml_backend_buffer_ptr buffer;

    impl(ggml_backend_t backend, uint32_t slots, const std::vector<expert_projection> & projections,
         uint64_t budget, std::function<expert_lease(expert_address)> source) :
        backend(backend), slots(slots), experts(0), source(std::move(source)) {
        if (!backend || !this->source || projections.size() != 3 || slots < 8 || slots > 512 || budget == 0 || budget > (1ull << 30)) {
            throw std::invalid_argument("invalid GLM expert bank backend, slots or budget");
        }
        if (projections.front().experts.size() < slots || projections.front().experts.size() > 512) {
            throw std::invalid_argument("expert bank cannot replace or invent model experts");
        }
        experts = uint32_t(projections.front().experts.size());
        context.reset(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
        if (!context) { throw std::runtime_error("cannot allocate expert bank tensor metadata"); }
        uint64_t logical = 0;
        for (size_t i = 0; i < projections.size(); ++i) {
            const auto & spec = projections[i];
            if (!bank_type(spec.type) ||
                    spec.input <= 0 || spec.input > 65536 || spec.output <= 0 || spec.output > 65536 ||
                    spec.input % ggml_blck_size(spec.type) || spec.experts.size() != experts) {
                throw std::invalid_argument("invalid expert bank quant/shape/catalog");
            }
            const uint64_t bytes = ggml_row_size(spec.type, spec.input) * uint64_t(spec.output);
            if (bytes == 0 || bytes > max_expert_read_bytes) { throw std::invalid_argument("expert bank slab exceeds source limits"); }
            if (bytes * slots > budget || logical > budget - bytes * slots) { throw std::invalid_argument("expert bank payload exceeds local budget"); }
            logical += bytes * slots;
            for (const auto & address : spec.experts) {
                if (address.file >= 100 || address.bytes != bytes || address.offset > (1ull << 44) - bytes) {
                    throw std::invalid_argument("expert bank descriptor differs from native quantized slab size/bounds");
                }
            }
            banks[i] = ggml_new_tensor_3d(context.get(), spec.type, spec.input, spec.output, slots);
            ggml_set_name(banks[i], ("glm-expert-bank-" + std::to_string(i)).c_str());
        }
        specs = projections;
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t planned = ggml_backend_alloc_ctx_tensors_from_buft_size(context.get(), buft);
        if (planned == 0 || planned > budget) { throw std::invalid_argument("expert bank backend allocation exceeds local budget"); }
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), buft));
        if (!buffer || ggml_backend_buffer_get_size(buffer.get()) > budget) { throw std::runtime_error("expert bank allocation failed or exceeded budget"); }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buffer.get(), 0);
    }

    ~impl() { ggml_backend_synchronize(backend); }
};

expert_bank::expert_bank(ggml_backend_t backend, uint32_t slots, const std::vector<expert_projection> & specs,
        uint64_t budget, std::function<expert_lease(expert_address)> source) :
    state(std::make_shared<impl>(backend, slots, specs, budget, std::move(source))) {}

uint64_t expert_bank::buffer_bytes() const { return ggml_backend_buffer_get_size(state->buffer.get()); }

expert_bank::lease::lease(std::shared_ptr<impl> state, std::vector<int32_t> ids) noexcept :
    state(std::move(state)), ids(std::move(ids)) {}

expert_bank::lease::~lease() {
    if (state) { ggml_backend_synchronize(state->backend); state->in_use.store(false); }
}

ggml_tensor * expert_bank::lease::projection(size_t index) const {
    if (!state || index >= state->banks.size()) { throw std::invalid_argument("invalid or expired expert bank lease"); }
    return state->banks[index];
}

const std::vector<int32_t> & expert_bank::lease::remapped_ids() const {
    if (!state) { throw std::runtime_error("expert bank lease was moved"); }
    return ids;
}

expert_bank::lease expert_bank::materialize(const std::vector<int32_t> & selected) {
    bool expected = false;
    if (!state->in_use.compare_exchange_strong(expected, true)) { throw std::runtime_error("expert bank has an active compute lease"); }
    try {
        if (state->failed) { throw std::runtime_error("expert bank is invalid; recreate its context"); }
        if (selected.empty() || selected.size() > 16384) { throw std::invalid_argument("expert bank selection count exceeds limits"); }
        std::array<int32_t, 512> mapping;
        mapping.fill(-1);
        std::vector<int32_t> unique, remapped;
        remapped.reserve(selected.size());
        for (const int32_t id : selected) {
            if (id < 0 || uint32_t(id) >= state->experts) { throw std::invalid_argument("selected expert ID is outside the model"); }
            if (mapping[id] < 0) {
                if (unique.size() == state->slots) { throw std::invalid_argument("distinct selected experts exceed the bank; no experts were clipped"); }
                mapping[id] = int32_t(unique.size());
                unique.push_back(id);
            }
            remapped.push_back(mapping[id]);
        }
        ggml_backend_synchronize(state->backend);
        for (size_t projection = 0; projection < state->specs.size(); ++projection) {
            const auto & spec = state->specs[projection];
            for (size_t slot = 0; slot < unique.size(); ++slot) {
                const auto & address = spec.experts[unique[slot]];
                const auto data = state->source(address);
                if (!data || data->size() != address.bytes) { throw std::runtime_error("expert source returned a missing or incorrectly sized slab"); }
                // tensor_set completes its host/device copy before this raw-source lease is released.
                ggml_backend_tensor_set(state->banks[projection], data->data(), slot * size_t(address.bytes), data->size());
            }
        }
        return lease(state, std::move(remapped));
    } catch (...) { state->failed = true; state->in_use.store(false); throw; }
}

} // namespace glm_stage
