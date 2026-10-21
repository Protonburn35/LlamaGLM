#ifndef LLAMA_STAGE_H
#define LLAMA_STAGE_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental owned-weight loading and local layer-stage evaluation.
struct llama_stage_model_params {
    uint32_t version;        // must be 1
    uint32_t rank;           // 0, 1 or 2
    uint32_t layer_ends[3];  // exclusive ends of three nonempty contiguous target ranges
};

// GLM5Next only; params.load_mode must be NONE (buffered), lazy_mode OFF, load_mtp/vocab_only false.
// Loads owned tensors into ordinary buffers. This does not establish Windows residency.
// Use llama_init_stage_from_model; ordinary context creation/model saving refuse partial models.
LLAMA_API struct llama_model * llama_model_load_stage_from_file(
        const char * path_model,
        struct llama_model_params params,
        struct llama_stage_model_params stage);

// Synthetic/custom tensor source equivalent, used for owned-weight qualification.
LLAMA_API struct llama_model * llama_model_init_stage_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params,
        struct llama_stage_model_params stage);

// Full model context by default, FP16 MLA/index caches, one sequence, no speculation.
LLAMA_API struct llama_context_params llama_stage_context_default_params(void);

// A/B output every four-stream FP32 residual row through llama_get_embeddings.
// B/C consume embedding rows at llama_model_n_embd_inp width; C outputs normal logits.
// Requires consecutive explicit positions within capacity, seq 0, no MTP/rollback/backend sampler, FP16 caches.
// A compute/preparation failure permanently invalidates this context; recreate it with the same loaded model.
LLAMA_API struct llama_context * llama_init_stage_from_model(
        struct llama_model * model,
        struct llama_context_params params);

#ifdef __cplusplus
}
#endif

#endif
