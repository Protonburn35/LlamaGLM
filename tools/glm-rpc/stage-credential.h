#pragma once

#include "stage-crypto.h"

#include <filesystem>
#include <memory>

namespace glm_rpc {

// Startup-only private local 32-byte file. The reader closes before construction returns.
// Native key pages have an explicit caller-reserved budget and no pageable fallback.
class locked_secret {
public:
    locked_secret(const std::filesystem::path & path, uint64_t page_budget);
    ~locked_secret();
    locked_secret(const locked_secret &) = delete;
    locked_secret & operator=(const locked_secret &) = delete;
    const secret & value() const;
    uint64_t locked_bytes() const;
    void verify_locked() const;
    // Quiescent only: join consumers before wiping borrowed key views.
    void invalidate() noexcept;
private:
    struct impl;
    std::unique_ptr<impl> state;
};

} // namespace glm_rpc
