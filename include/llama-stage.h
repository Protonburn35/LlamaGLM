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

struct llama_stage_storage_params {
    uint32_t version;              // must be 1
    uint32_t bank_slots;           // at least all selected experts; no clipping or implicit waves
    uint64_t host_cache_bytes;     // raw desktop expert payload reservation
    uint64_t context_bank_bytes;   // aggregate backend bank allocation cap per context
    uint32_t host_cache_entries;   // 1..4096
    uint32_t max_active_reads;     // 1..8
};

struct llama_stage_storage_report {
    uint32_t version; // caller initializes to 1
    uint32_t streamed_tensors;
    uint64_t routed_logical_bytes;
    uint64_t model_buffer_bytes;
    uint64_t host_reserved_bytes;
    uint64_t native_read_requests;
    uint64_t native_system_reads;
    uint64_t native_read_bytes;
};

LLAMA_API bool llama_model_stage_storage_report(const struct llama_model * model, struct llama_stage_storage_report * report);

struct llama_stage_bank_report {
    uint32_t version; // caller initializes to 1; inspect between decodes
    uint32_t cpu_banks;
    uint32_t device_banks;
    uint32_t reserved;
    uint64_t buffer_bytes;
    uint64_t materializations;
    uint64_t slot_hits;   // distinct expert groups in successful materializations
    uint64_t slot_misses;
    uint64_t copied_slabs;
    uint64_t copied_bytes;
    uint64_t failures;
};

// Backend slot/copy counters, not physical GPU residency or admission.
LLAMA_API bool llama_context_stage_bank_report(const struct llama_context * ctx, struct llama_stage_bank_report * report);

struct llama_stage_preload_params {
    uint32_t version; // 1: locked experts; 2: shared locked experts and CPU tensor buffers
    uint32_t bank_slots;
    uint64_t host_payload_bytes;
    uint64_t host_page_bytes; // v2: combined expert and CPU tensor page budget
    uint64_t context_bank_bytes;
    uint32_t host_entries; // 1..100000, complete owned routed catalog
    uint32_t reserved;     // must be 0
};

struct llama_stage_preload_report {
    uint32_t version; // caller initializes to 1
    uint32_t expert_slabs;
    uint32_t open_source_files;
    uint32_t full_residency_validated; // remains 0: other weights/state/GPU admission is separate
    uint64_t payload_bytes;
    uint64_t locked_bytes;
    uint64_t startup_read_requests;
    uint64_t startup_read_bytes;
};

LLAMA_API bool llama_model_stage_preload_report(const struct llama_model * model, struct llama_stage_preload_report * report);

struct llama_stage_host_report {
    uint32_t version; // caller initializes to 1; inspect between decodes and allocation/free operations
    uint32_t native_payload_locks_verified;
    uint32_t full_residency_validated; // always 0: backend work/metadata/network/GPU admission is separate
    int32_t last_native_error;
    uint64_t page_budget;
    uint64_t reserved_page_bytes;
    uint64_t peak_reserved_page_bytes;
    uint64_t expert_page_bytes;
    uint64_t tensor_requested_bytes;
    uint64_t tensor_buffers;
    uint64_t pending_buffers;
    uint64_t allocations;
    uint64_t refusals;
    uint64_t failures;
};

// Available only for preload policy 2. Verifies the tracked native payload pages, not full admission.
LLAMA_API bool llama_model_stage_host_report(const struct llama_model * model, struct llama_stage_host_report * report);

struct llama_stage_schema_report {
    uint32_t version; // caller must initialize to 1
    uint32_t tensors;
    uint32_t target_layers;
    uint32_t nextn_layers;
    uint64_t payload_bytes; // logical bytes only; no payload is allocated or read
};

// Validate a complete GLM5Next descriptor table, including declared NextN, against the native loader.
// Uses metadata and zero-sized weight buffers only. Does not verify payloads, identity or residency.
LLAMA_API bool llama_model_validate_stage_schema(
        struct gguf_context * metadata,
        struct llama_stage_schema_report * report);

// GLM5Next only; params.load_mode must be NONE (buffered), lazy_mode OFF, load_mtp/vocab_only false.
// Loads owned tensors into ordinary buffers. This does not establish Windows residency.
// Use llama_init_stage_from_model; ordinary context creation/model saving refuse partial models.
LLAMA_API struct llama_model * llama_model_load_stage_from_file(
        const char * path_model,
        struct llama_model_params params,
        struct llama_stage_model_params stage);

// Experimental desktop ranks 0/2 only; requires LLAMA_GLM_RPC and buffered NONE/lazy OFF.
// Dense/shared weights load normally. Routed expert descriptors remain unallocated until bank fills.
// Does not establish resource admission or residency. No MTP/LoRA/tensor split/full-payload check.
LLAMA_API struct llama_model * llama_model_load_stage_streamed_from_file(
        const char * path_model,
        struct llama_model_params params,
        struct llama_stage_model_params stage,
        struct llama_stage_storage_params storage);

// Experimental rank 1: preload every owned routed expert into locked host pages and close its readers.
// v1 protects expert payloads; v2 also routes ordinary host tensor/cache/graph/output/bank
// buffers through the same page budget. v2 requires use_extra_bufts=false and no buffer overrides.
// Backend work buffers, metadata, network/staging and GPU residency still require admission.
LLAMA_API struct llama_model * llama_model_load_stage_preloaded_from_file(
        const char * path_model,
        struct llama_model_params params,
        struct llama_stage_model_params stage,
        struct llama_stage_preload_params preload);

// Synthetic/custom tensor source equivalent, used for owned-weight qualification.
LLAMA_API struct llama_model * llama_model_init_stage_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params,
        struct llama_stage_model_params stage);

// Full model context by default, FP16 MLA/index caches, one sequence, no speculation.
LLAMA_API struct llama_context_params llama_stage_context_default_params(void);

// Same context/precision defaults, with an explicit microbatch fitting this model's expert banks.
LLAMA_API struct llama_context_params llama_stage_context_default_params_for_model(const struct llama_model * model);

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
