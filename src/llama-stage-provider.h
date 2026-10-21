#pragma once

#include "llama-stage.h"
#include "llama-arch.h"
#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;

struct llama_stage_context_experts {
    virtual ~llama_stage_context_experts() = default;
    virtual bool owns(const ggml_tensor * tensor) const = 0;
    virtual ggml_tensor * projection(const ggml_tensor * tensor) const = 0;
    virtual ggml_tensor * build_slots(ggml_context * ctx, ggml_tensor * selected, int layer) = 0;
    virtual bool failed() const = 0;
    virtual std::string error() const = 0;
    virtual void report(llama_stage_bank_report & output) const = 0;
    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;
};

struct llama_stage_weight_source {
    virtual ~llama_stage_weight_source() = default;
    virtual ggml_tensor * register_tensor(const LLM_TN_IMPL & name, const ggml_tensor * descriptor, uint16_t file, uint64_t offset) = 0;
    virtual void finalize(const std::vector<std::string> & paths) = 0;
    virtual void append_tensors(std::vector<std::pair<std::string, ggml_tensor *>> & output) const = 0;
    virtual bool empty() const = 0;
    virtual uint32_t slots() const = 0;
    virtual void report(llama_stage_storage_report & output) const = 0;
    virtual bool report_preload(llama_stage_preload_report & output) const = 0;
    virtual bool report_host(llama_stage_host_report & output) const = 0;
    virtual ggml_backend_buffer_type_t host_buffer_type(ggml_backend_buffer_type_t original) const = 0;
    virtual void configure_backend(ggml_backend_t backend) const = 0;
    virtual std::unique_ptr<llama_stage_context_experts> create_context(const std::vector<ggml_backend_t> & backends) = 0;
};

int llama_stage_expert_projection(const LLM_TN_IMPL & name);
std::shared_ptr<llama_stage_weight_source> llama_stage_make_weight_source(
        const llama_model & model, const llama_stage_storage_params & params);
std::shared_ptr<llama_stage_weight_source> llama_stage_make_preloaded_source(
        const llama_model & model, const llama_stage_preload_params & params);
