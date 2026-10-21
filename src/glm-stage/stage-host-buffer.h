#pragma once

#include "stage-locked-pages.h"
#include "ggml-backend.h"

#include <memory>

namespace glm_stage {

struct host_buffer_stats {
    uint64_t reserved_page_bytes, peak_reserved_page_bytes, requested_bytes;
    size_t reserved_buffers, pending_buffers;
    uint64_t allocations, refusals, failures;
    int last_native_error;
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
private:
    std::shared_ptr<impl> state;
};

} // namespace glm_stage
