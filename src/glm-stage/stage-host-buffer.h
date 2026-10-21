#pragma once

#include "stage-locked-pages.h"
#include "ggml-backend.h"

#include <memory>
#include <functional>

namespace glm_stage {

struct host_buffer_stats {
    uint64_t reserved_page_bytes, peak_reserved_page_bytes, requested_bytes;
    size_t reserved_buffers, pending_buffers;
    uint64_t allocations, refusals, failures;
    int last_native_error;
    uint64_t external_page_bytes;
    bool external_pending;
};

// Standard CPU-layout data buffers with native locks. Wait for backend work before freeing.
// Nonempty buffers retain the allocator; GGML zero-size dummy buffers borrow the pool's lifetime.
class host_buffer_pool {
    struct impl;
public:
    host_buffer_pool(ggml_backend_dev_t cpu, uint64_t page_budget, size_t buffer_budget = 4096);
    ggml_backend_buffer_type_t buffer_type() const;
    host_buffer_stats stats() const;
    void verify_all() const; // between allocation/free operations
    // Startup-only: reserve before loading, retain the returned owner, never retain the loader.
    // The verifier must check native locks and return the exact reserved page count without
    // calling this pool. The owner must not retain the pool or its GGML buffers.
    std::shared_ptr<void> preload_external_pages(uint64_t page_bytes,
        const std::function<std::shared_ptr<void>()> & loader, uint64_t (*verify)(const void *));
private:
    std::shared_ptr<impl> state;
};

} // namespace glm_stage
