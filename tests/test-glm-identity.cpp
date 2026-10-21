#include "stage-identity.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <stdexcept>

using namespace glm_rpc;

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void refused(const std::function<void()> & action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid identity was accepted");
}

static std::string hex(const digest & value) {
    const char digits[] = "0123456789abcdef";
    std::string result;
    for (const uint8_t byte : value) { result += digits[byte >> 4]; result += digits[byte & 15]; }
    return result;
}

static execution_identity fixture() {
    execution_identity value;
    value.source_revision = std::string(40, '1');
    value.model_repository = "test/GLM";
    value.model_revision = std::string(40, '2');
    value.quant = "synthetic";
    digest a, b; a.fill(3); b.fill(4);
    value.shards = {{0, "fixture-1.gguf", 4096, 64, a}, {1, "fixture-2.gguf", 8192, 96, b}};
    value.tensors = 17; value.payload_bytes = 1000;
    value.target_layers = 6; value.nextn_layers = 0; value.experts = 16;
    value.layer_ends = {2, 4, 6};
    value.bounds.hidden = 128; value.bounds.rows = 64; value.bounds.context = 256; value.bounds.frame_bytes = 65536;
    return value;
}

static void crypto_vectors() {
    require(hex(sha256({})) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty SHA256 vector failed");
    const uint8_t abc[] = "abc";
    const auto expected = sha256({{abc, 3}});
    require(hex(expected) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA256 abc vector failed");
    require(equal_digest(expected, sha256({{abc, 1}, {nullptr, 0}, {abc + 1, 2}})), "multipart SHA256 differs");
    refused([] { sha256({{nullptr, 1}}); });
}

static void canonical_vector() {
    auto value = fixture();
    const auto expected = identity_digest(value);
    // Independent Python struct.pack/hashlib vector for the documented version-1 field order.
    require(hex(expected) == "9ac22298ddd0f1d758a779b38c126ccfab9270ce61036b0ddd852bb7435b6f64", "canonical identity vector failed");
    std::reverse(value.shards.begin(), value.shards.end());
    require(equal_digest(expected, identity_digest(value)), "shard enumeration order changed canonical identity");
}

static void field_binding() {
    const auto original = fixture();
    const auto expected = identity_digest(original);
    const std::vector<std::function<void(execution_identity &)>> changes{
        [](auto & v) { v.source_revision[0] = '3'; }, [](auto & v) { v.model_revision[0] = '3'; },
        [](auto & v) { v.model_repository += "x"; }, [](auto & v) { v.quant += "x"; },
        [](auto & v) { ++v.tensors; }, [](auto & v) { ++v.payload_bytes; },
        [](auto & v) { v.target_layers = v.layer_ends[2] = 7; }, [](auto & v) { v.nextn_layers = 1; },
        [](auto & v) { ++v.experts; }, [](auto & v) { ++v.experts_used; }, [](auto & v) { ++v.shared_experts; },
        [](auto & v) { ++v.layer_ends[0]; }, [](auto & v) { --v.layer_ends[1]; },
        [](auto & v) { ++v.bounds.hidden; }, [](auto & v) { ++v.bounds.rows; }, [](auto & v) { ++v.bounds.context; },
        [](auto & v) { v.bounds.frame_bytes *= 2; }, [](auto & v) { ++v.shards[0].file_bytes; },
        [](auto & v) { ++v.shards[0].header_bytes; }, [](auto & v) { v.shards[0].header_sha256[0] ^= 1; },
        [](auto & v) { v.shards[0].name = "other.gguf"; },
    };
    for (const auto & change : changes) {
        auto value = original; change(value);
        require(!equal_digest(expected, identity_digest(value)), "execution field is not bound to identity");
    }
}

static void invalid_contracts() {
    const std::vector<std::function<void(execution_identity &)>> changes{
        [](auto & v) { v.source_revision = "unknown"; }, [](auto & v) { v.model_revision[0] = 'A'; },
        [](auto & v) { v.quant = "../quant"; }, [](auto & v) { v.model_repository += '\n'; },
        [](auto & v) { v.shards.clear(); }, [](auto & v) { v.shards.resize(101); },
        [](auto & v) { v.shards[1].number = 0; }, [](auto & v) { v.shards[1].number = 2; },
        [](auto & v) { v.shards[1].name = "FIXTURE-1.gguf"; }, [](auto & v) { v.shards[0].name = "C:\\model.gguf"; },
        [](auto & v) { v.shards[0].header_sha256.fill(0); }, [](auto & v) { v.shards[0].header_bytes = (32ull << 20) + 1; },
        [](auto & v) { v.shards[0].file_bytes = 1; }, [](auto & v) { v.payload_bytes = 1000000; },
        [](auto & v) { v.layer_ends[0] = v.layer_ends[1]; }, [](auto & v) { --v.layer_ends[2]; },
        [](auto & v) { v.bounds.streams = 1; }, [](auto & v) { v.bounds.context = 1048577; },
        [](auto & v) { v.tensor_schema = 2; }, [](auto & v) { v.state_layout = 2; },
        [](auto & v) { v.experts_used = 17; }, [](auto & v) { v.nextn_layers = 2; },
    };
    for (const auto & change : changes) {
        auto value = fixture(); change(value);
        refused([&] { identity_digest(value); });
    }
}

int main() {
    try {
        crypto_vectors(); canonical_vector(); field_binding(); invalid_contracts();
        std::puts("GLM identity: four suites passed (SHA256, canonical vector, field binding, malformed contracts)");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM identity test failed: %s\n", error.what());
        return 1;
    }
}
