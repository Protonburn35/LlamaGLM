#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace glm_rpc {

using digest = std::array<uint8_t, 32>;
using secret = std::array<uint8_t, 32>;

struct bytes_view {
    const uint8_t * data;
    size_t size;
};

digest hmac_sha256(const secret & key, const std::vector<bytes_view> & parts);
digest sha256(const std::vector<bytes_view> & parts);
digest random_nonce();
bool equal_digest(const digest & a, const digest & b) noexcept;
void erase_secret(secret & value) noexcept;

} // namespace glm_rpc
