#ifndef LLAMA_STAGE_H
#define LLAMA_STAGE_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental owned-weight loading. This API does not yet provide a stage evaluator.
struct llama_stage_model_params {
    uint32_t version;        // must be 1
    uint32_t rank;           // 0, 1 or 2
    uint32_t layer_ends[3];  // exclusive ends of three nonempty contiguous target ranges
};

// GLM5Next only; params.load_mode must be NONE (buffered), lazy_mode OFF, load_mtp/vocab_only false.
// Loads owned tensors into ordinary buffers. This does not establish Windows residency.
// The returned partial model must not be used with llama_init_from_model or model saving.
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

#ifdef __cplusplus
}
#endif

#endif
