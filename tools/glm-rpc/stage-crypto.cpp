#include "stage-crypto.h"

#include <climits>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#else
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

namespace glm_rpc {

digest hmac_sha256(const secret & key, const std::vector<bytes_view> & parts) {
    for (const auto & part : parts) {
        if ((!part.data && part.size) || part.size > INT_MAX) {
            throw std::invalid_argument("invalid authentication input");
        }
    }
    digest result{};
#ifdef _WIN32
    struct provider {
        BCRYPT_ALG_HANDLE value = nullptr;
        ~provider() { if (value) { BCryptCloseAlgorithmProvider(value, 0); } }
    } algorithm;
    if (BCryptOpenAlgorithmProvider(&algorithm.value, BCRYPT_SHA256_ALGORITHM, nullptr, BCRYPT_ALG_HANDLE_HMAC_FLAG) != 0) {
        throw std::runtime_error("cannot initialize SHA256 authentication provider");
    }
    struct hash {
        BCRYPT_HASH_HANDLE value = nullptr;
        ~hash() { if (value) { BCryptDestroyHash(value); } }
    } state;
    if (BCryptCreateHash(algorithm.value, &state.value, nullptr, 0,
            const_cast<PUCHAR>(key.data()), ULONG(key.size()), 0) != 0) {
        throw std::runtime_error("cannot initialize authentication state");
    }
    for (const auto & part : parts) {
        if (part.size && BCryptHashData(state.value, const_cast<PUCHAR>(part.data), ULONG(part.size), 0) != 0) {
            throw std::runtime_error("cannot authenticate message");
        }
    }
    if (BCryptFinishHash(state.value, result.data(), ULONG(result.size()), 0) != 0) {
        throw std::runtime_error("cannot finish message authentication");
    }
#else
    std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> private_key(
        EVP_PKEY_new_raw_private_key(EVP_PKEY_HMAC, nullptr, key.data(), key.size()), &EVP_PKEY_free);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> state(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (!private_key || !state || EVP_DigestSignInit(state.get(), nullptr, EVP_sha256(), nullptr, private_key.get()) != 1) {
        throw std::runtime_error("cannot initialize SHA256 authentication state");
    }
    for (const auto & part : parts) {
        if (part.size && EVP_DigestSignUpdate(state.get(), part.data, part.size) != 1) {
            throw std::runtime_error("cannot authenticate message");
        }
    }
    size_t size = result.size();
    if (EVP_DigestSignFinal(state.get(), result.data(), &size) != 1 || size != result.size()) {
        throw std::runtime_error("cannot finish message authentication");
    }
#endif
    return result;
}

digest random_nonce() {
    digest result{};
#ifdef _WIN32
    if (BCryptGenRandom(nullptr, result.data(), ULONG(result.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
#else
    if (RAND_bytes(result.data(), int(result.size())) != 1) {
#endif
        throw std::runtime_error("system random number generation failed");
    }
    return result;
}

bool equal_digest(const digest & a, const digest & b) noexcept {
    volatile uint8_t difference = 0;
    for (size_t i = 0; i < a.size(); ++i) { difference = uint8_t(difference | (a[i] ^ b[i])); }
    return difference == 0;
}

void erase_secret(secret & value) noexcept {
#ifdef _WIN32
    SecureZeroMemory(value.data(), value.size());
#else
    OPENSSL_cleanse(value.data(), value.size());
#endif
}

} // namespace glm_rpc
