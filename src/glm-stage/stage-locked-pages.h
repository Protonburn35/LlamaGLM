#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace glm_stage {

constexpr uint64_t max_locked_buffer_bytes = 1ull << 40;

struct locked_buffer_stats {
    uint64_t bytes;
    uint64_t buffers;
};

// Dedicated native pages. The caller must reserve aggregate capacity before construction.
class locked_pages {
public:
    locked_pages(size_t bytes, uint64_t page_budget);
    ~locked_pages();
    locked_pages(const locked_pages &) = delete;
    locked_pages & operator=(const locked_pages &) = delete;
    uint8_t * data();
    const uint8_t * data() const;
    size_t size() const;
    uint64_t locked_bytes() const;
    void verify_locked() const;
    static uint64_t required_bytes(size_t bytes);
    static locked_buffer_stats locked_stats();
private:
    struct impl;
    std::unique_ptr<impl> state;
};

} // namespace glm_stage
