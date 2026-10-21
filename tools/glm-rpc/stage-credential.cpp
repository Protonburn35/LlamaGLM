#include "stage-credential.h"
#include "stage-storage.h"

#include <algorithm>
#include <new>
#include <stdexcept>

namespace glm_rpc {
namespace {
struct key_pages {
    glm_stage::locked_pages pages;
    explicit key_pages(uint64_t budget) : pages(sizeof(secret),budget) { new (pages.data()) secret{}; }
    ~key_pages() { erase_secret(value()); }
    secret & value() { return *reinterpret_cast<secret *>(pages.data()); }
};
}

struct locked_secret::impl {
    key_pages key;
    bool valid = false;
    impl(const std::filesystem::path & path, uint64_t budget) : key(budget) {
        glm_stage::expert_file input(path);
        if (input.size() != sizeof(secret)) { throw std::invalid_argument("credential file must contain exactly 32 bytes"); }
        input.require_private_access();
        input.read_exact(0,key.value().data(),key.value().size());
        input.require_private_access();
        if (std::all_of(key.value().begin(),key.value().end(),[](uint8_t byte) { return byte == 0; })) {
            throw std::invalid_argument("credential file must contain a nonzero configured key");
        }
        key.pages.verify_locked(); valid = true;
    }
};

locked_secret::locked_secret(const std::filesystem::path & path, uint64_t budget) : state(std::make_unique<impl>(path,budget)) {}
locked_secret::~locked_secret() = default;
const secret & locked_secret::value() const {
    if (!state->valid) { throw std::logic_error("configured key was invalidated"); }
    return state->key.value();
}
uint64_t locked_secret::locked_bytes() const { return state->key.pages.locked_bytes(); }
void locked_secret::verify_locked() const { value(); state->key.pages.verify_locked(); }
void locked_secret::invalidate() noexcept { state->valid = false; erase_secret(state->key.value()); }

} // namespace glm_rpc
