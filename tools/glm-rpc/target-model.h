#pragma once

#include <array>
#include <cstdint>

namespace glm_rpc { namespace target_model {

inline constexpr const char * repository = "unsloth/GLM-5.3-Flash-GGUF";
inline constexpr const char * revision = "a38483c8cd5df544f53d70fb281afe97369d5ab6";
inline constexpr const char * quant = "UD-IQ3_XXS";
inline constexpr uint32_t tensors = 1412;
inline constexpr uint64_t payload_bytes = 120358051192ull;

struct shard {
    const char * name;
    uint64_t file_bytes, header_bytes;
    const char * header_sha256;
};

inline constexpr std::array<shard,4> shards{{
    {"GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf",9429984,9429956,"e47e6ba043a5d406cdf695219ff9728666dc3a0392eb7190171ca13fd671d63d"},
    {"GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00004.gguf",49261511584ull,38304,"ad6191c93269faeee16be8d2d1746df56f43e7f6515c26d646368a358f0ad7d4"},
    {"GLM-5.3-Flash-UD-IQ3_XXS-00003-of-00004.gguf",49075832480ull,35232,"eed639822ed7b1e16efe6a4ab6d68728b5b3648c3270fe379218fb984097747e"},
    {"GLM-5.3-Flash-UD-IQ3_XXS-00004-of-00004.gguf",22020797792ull,15328,"269b5ce802b9e424ad10d9781c709eace77e3310e5e1ba63939a3c074bfb20e3"},
}};

} } // namespace glm_rpc::target_model
