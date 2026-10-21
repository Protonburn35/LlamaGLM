#include "stage-credential.h"
#include "stage-storage.h"
#include "stage-wire.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>

using namespace glm_rpc;
static void require(bool value, const char * reason) { if (!value) { throw std::runtime_error(reason); } }
template<class F> static void refused(F action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid credential policy was accepted");
}

static int run(const std::array<std::filesystem::path,4> & paths) {
    try {
        const auto baseline = glm_stage::locked_pages::locked_stats();
        const auto budget = glm_stage::locked_pages::required_bytes(sizeof(secret));
        refused([&] { locked_secret key(paths[0],budget-1); });
        for (size_t i = 1; i < paths.size(); ++i) { refused([&] { locked_secret key(paths[i],budget); }); }
        auto current = glm_stage::locked_pages::locked_stats();
        require(current.bytes == baseline.bytes && current.buffers == baseline.buffers,"failed credential load retained native page charges");
        {
            glm_stage::expert_file file(paths[1]);
            refused([&] { file.require_private_access(); });
            secret scratch{}; refused([&] { file.read_exact(0,scratch.data(),scratch.size()); });
        }
        {
            locked_secret first(paths[0],budget), second(paths[0],budget);
            require(first.locked_bytes() == budget && second.locked_bytes() == budget,"credential lock charge is inconsistent");
            first.verify_locked(); second.verify_locked();
            secret expected{};
            for (size_t i = 0; i < expected.size(); ++i) { expected[i] = uint8_t(i+1); }
            require(first.value() == expected && second.value() == expected,"private native key bytes changed");
            std::error_code error;
            require(std::filesystem::remove(paths[0],error) && !error,"credential retained an inference file handle");
            refused([&] { locked_secret missing(paths[0],budget); });
            binding peer; peer.server_nonce.fill(1); peer.client_nonce.fill(2); peer.identity.fill(3);
            const auto derived = derive_link_key(expected,peer);
            std::vector<std::thread> readers;
            std::atomic<bool> matched{true};
            for (size_t i = 0; i < 8; ++i) {
                readers.emplace_back([&] {
                    for (size_t j = 0; j < 100; ++j) {
                        if (derive_link_key(first.value(),peer) != derived) { matched.store(false); }
                    }
                });
            }
            for (auto & reader : readers) { reader.join(); }
            require(matched.load(),"source-deleted concurrent native key use changed its authentication result");
            const auto & borrowed = first.value();
            first.invalidate(); first.invalidate();
            require(std::all_of(borrowed.begin(),borrowed.end(),[](uint8_t byte) { return byte == 0; }),"invalidated credential did not wipe borrowed bytes");
            refused([&] { first.value(); }); refused([&] { first.verify_locked(); });
            require(second.value() == expected && first.locked_bytes() == budget,"one key invalidation changed another native owner or released a live charge");
            current = glm_stage::locked_pages::locked_stats();
            require(current.bytes == baseline.bytes+2*budget && current.buffers == baseline.buffers+2,"credential page lifetime accounting differs");
            erase_secret(expected);
        }
        current = glm_stage::locked_pages::locked_stats();
        require(current.bytes == baseline.bytes && current.buffers == baseline.buffers,"credential destruction retained native pages");
        std::puts("GLM credentials: six suites passed (private access, size/nonzero/cap refusal, native pages/cleanup, closed source, eight readers, quiescent wipe); full process admission NOT_VALIDATED");
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr,"GLM credential test failed: %s\n",error.what()); return 1; }
}

#ifdef _WIN32
int wmain(int argc, wchar_t ** argv) {
    if (argc != 5) { return 2; }
    return run({argv[1],argv[2],argv[3],argv[4]});
}
#else
int main(int argc, char ** argv) {
    if (argc != 5) { return 2; }
    return run({argv[1],argv[2],argv[3],argv[4]});
}
#endif
