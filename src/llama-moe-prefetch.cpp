#include "llama-moe-prefetch.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

struct prefetch_range {
    const char * addr;
    size_t       size;
};

std::atomic<uint64_t> g_calls{0};
std::atomic<uint64_t> g_bytes{0};
std::atomic<int64_t>  g_t_report{0};

void prefetch_ranges(const std::vector<prefetch_range> & ranges) {
#ifdef _WIN32
#if _WIN32_WINNT >= 0x602
    using prefetch_fn_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const prefetch_fn_t fn = (prefetch_fn_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    if (fn == nullptr) {
        return;
    }
    std::vector<WIN32_MEMORY_RANGE_ENTRY> entries(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        entries[i].VirtualAddress = (PVOID) ranges[i].addr;
        entries[i].NumberOfBytes  = (SIZE_T) ranges[i].size;
    }
    // a hint: a failure only means the pages are faulted in as usual
    fn(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0);
#else
    GGML_UNUSED(ranges);
#endif
#else
    static const uintptr_t page_size = (uintptr_t) sysconf(_SC_PAGESIZE);
    for (const auto & r : ranges) {
        const uintptr_t beg = (uintptr_t) r.addr & ~(page_size - 1);
        const uintptr_t end = (uintptr_t) r.addr + r.size;
        posix_madvise((void *) beg, end - beg, POSIX_MADV_WILLNEED);
    }
#endif
}

// src[0]: selected experts [n_expert_used, n_tokens], src[1..]: expert tensors [ne0, ne1, n_expert]
void prefetch_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }

    const ggml_tensor * ids = dst->src[0];
    const int64_t n_expert = dst->src[1]->ne[2];

    std::vector<bool> used(n_expert, false);
    int64_t n_used = 0;
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t j = 0; j < ids->ne[0]; ++j) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + j*ids->nb[0] + t*ids->nb[1]);
            if (e >= 0 && e < n_expert && !used[e]) {
                used[e] = true;
                n_used++;
            }
        }
    }

    // large batches select most experts, read whole tensors then
    const bool whole = 2*n_used >= n_expert;

    std::vector<prefetch_range> ranges;
    size_t bytes = 0;
    for (int i = 1; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        const ggml_tensor * w = dst->src[i];
        const char * base = (const char *) w->data;
        if (whole) {
            ranges.push_back({ base, ggml_nbytes(w) });
            bytes += ggml_nbytes(w);
            continue;
        }
        // consecutive selected experts are one range
        for (int64_t e0 = 0; e0 < n_expert; ) {
            if (!used[e0]) {
                e0++;
                continue;
            }
            int64_t e1 = e0 + 1;
            while (e1 < n_expert && used[e1]) {
                e1++;
            }
            const size_t size = (size_t) (e1 - e0)*w->nb[2];
            ranges.push_back({ base + e0*w->nb[2], size });
            bytes += size;
            e0 = e1;
        }
    }

    prefetch_ranges(ranges);

    const uint64_t calls = ++g_calls;
    const uint64_t total = g_bytes += bytes;
    const int64_t  now   = ggml_time_us();
    int64_t last = g_t_report.load();
    if (now - last >= 30*1000000LL && g_t_report.compare_exchange_strong(last, now)) {
        // the first call is reported too, as confirmation that LLAMA_MOE_PREFETCH is active
        LLAMA_LOG_INFO("llama_moe_prefetch: %llu calls, %.2f GiB of experts requested (LLAMA_MOE_PREFETCH)\n",
            (unsigned long long) calls, total/1024.0/1024.0/1024.0);
    }
}

bool is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

bool llama_moe_prefetch_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_MOE_PREFETCH");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

ggml_tensor * llama_moe_prefetch(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * const * exps, int n_exps) {
    if (!llama_moe_prefetch_enabled() || ids->type != GGML_TYPE_I32) {
        return nullptr;
    }

    std::vector<ggml_tensor *> args = { ids };
    for (int i = 0; i < n_exps; ++i) {
        ggml_tensor * w = exps[i];
        if (w == nullptr) {
            continue;
        }
        if (!is_host_weight(w) || (args.size() > 1 && w->ne[2] != args[1]->ne[2])) {
            return nullptr;
        }
        args.push_back(w);
    }
    if (args.size() < 2 || args.size() > GGML_MAX_SRC) {
        return nullptr;
    }

    return ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args.data(), (int) args.size(), prefetch_op, 1, nullptr);
}
