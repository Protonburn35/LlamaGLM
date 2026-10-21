#include "stage-storage.h"

#include <stdexcept>

namespace glm_stage {

struct expert_buffer::impl {
    size_t bytes;
    std::unique_ptr<uint8_t[]> heap;
    std::unique_ptr<locked_pages> pages;
    impl(size_t count, bool lock) : bytes(count) {
        if (bytes == 0 || bytes > max_expert_read_bytes) { throw std::invalid_argument("invalid expert payload allocation size"); }
        if (lock) { pages = std::make_unique<locked_pages>(bytes, locked_pages::required_bytes(bytes)); }
        else { heap = std::make_unique<uint8_t[]>(bytes); }
    }
    uint8_t * data() const { return pages ? pages->data() : heap.get(); }
};

expert_buffer::expert_buffer(size_t bytes, bool lock_pages) : state(std::make_unique<impl>(bytes, lock_pages)) {}
expert_buffer::~expert_buffer() = default;
uint8_t * expert_buffer::data() { return state->data(); }
const uint8_t * expert_buffer::data() const { return state->data(); }
size_t expert_buffer::size() const { return state->bytes; }
uint64_t expert_buffer::locked_bytes() const { return state->pages ? state->pages->locked_bytes() : 0; }
locked_buffer_stats expert_buffer::locked_stats() { return locked_pages::locked_stats(); }
uint64_t expert_buffer::required_locked_bytes(size_t bytes) {
    if (bytes == 0 || bytes > max_expert_read_bytes) { throw std::invalid_argument("invalid expert payload allocation size"); }
    return locked_pages::required_bytes(bytes);
}
void expert_buffer::verify_locked() const {
    if (!state->pages) { throw std::runtime_error("expert payload has no native page lock"); }
    state->pages->verify_locked();
}

} // namespace glm_stage
