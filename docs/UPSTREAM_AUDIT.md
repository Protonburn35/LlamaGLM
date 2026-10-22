# Upstream audit

This document was imported from the v0.8.0 VM snapshot. Commit names below describe that source history; the GitHub import and follow-up checks are recorded in `REVIEW-v0.8.0.md`.

## Pinned revisions

| What | Revision |
|---|---|
| Engine base | llama.cpp tag **b11780** = `1242be66f` ("server: add realtime transcription API (#30636)", 2026-10-21). `src/ ggml/ common/ tools/ include/` of `LlamaGLM-master` commit `b9a5776` ("baseline") are byte-identical to it (git tree hashes). |
| Fork `github.com/Protonburn35/LlamaGLM` `master` | `1242be66f` (= upstream, no fork changes) |
| Fork branch `codex/glm53-3pc-rpc` (Sol's custom staged engine) | `bc9748c2d`, 56 commits on `1242be66f`; **not** part of this tree |
| Upstream `master` checked | `a37cc4610` (2026-10-21 23:26 +0200), 23 commits after b11780 |
| Official Windows CUDA binaries of the base | release `b11780`: `llama-b11780-bin-win-cuda-13.4-x64.zip` (+ `cudart-llama-bin-win-cuda-13.4-x64.zip`), built with nvcc/MSVC, default arch list incl. `89-real` and `120a-real` |

So "LlamaGLM" = upstream llama.cpp b11780 + the small patch set below. GLM-5.3-Flash support (KDA, DSA with k-pool indexer, 4-stream hyper-connections, MoE, NextN/MTP) is upstream code.

## GLM5-next related upstream PRs present in b11780 (titles from the upstream log)

| PR | Upstream title |
|---|---|
| #27773 | add GLM-5.3-Flash (GLM5-Next) support |
| #29928 | feat: add GLM5Next MTP, optimize |
| #29958 | llama : fix unexpected graph reallocation in the k-pool models |
| #30042 | llama: remove the gather path of the glm5-next sparse attention |
| #29887 | llama : add a GPU cache for MoE experts kept in host memory (`--moe-cache-mib`; works only for experts in local host memory, not used by LlamaGLM) |
| #30112 | llama: support MoE cache over multiple GPUs |
| #30637 | moe-cache : replace MoE LRU cache with a 50/50 SLRU cache |

(The previous version of this file described several of these wrongly, e.g. #30042 as a fused DSA CUDA kernel.)

Verified on the VM: the loader of b11780 accepts the real `UD-IQ3_XXS` metadata (1412 tensors, `glm5-next`, 46 blocks) and builds the graph for the full model (test T8).

## Upstream changes after b11780 (to a37cc4610) that touch relevant code

| Commit | Subject | Decision |
|---|---|---|
| `8baabd53b` | scale moe down projection per expert token (#30861) | precision refinement of an existing rescale; not adopted, so that our build and the official b11780 binaries stay comparable |
| `d454a204e` | create new fused op for QSA mask (#30640), also used by glm5-next | performance; not adopted for the same reason |
| `7775ffb7e`, `a37cc4610` | new models | irrelevant |

Revisit after the first successful run: moving to a newer upstream tag is easy because the LlamaGLM delta is small.

## LlamaGLM delta vs b11780 (`git diff b9a5776 HEAD` in `LlamaGLM-master`)

| File | Change | Why |
|---|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` | define `THREAD_POWER_THROTTLING_*` if the MinGW headers lack them | MinGW cross build |
| `ggml/src/ggml-cuda/CMakeLists.txt` | clang CUDA flags (device fast math **without** finite-math, CCCL shim, emulated TLS); nvcc-only options skipped for clang | clang cross build |
| `ggml/src/ggml-rpc/ggml-rpc.cpp`, `ggml/include/ggml-rpc.h` | worker `[stats]` and `GGML_RPC_SERVER_RESERVE_MIB`; client byte counters + proc addresses for `llama-rpc-test` | proof of remote work, worker RAM/VRAM safety |
| `common/fit.cpp`, `fit.h`, `common.cpp`, `common.h`, `arg.cpp`, `tools/fit-params/fit-params.cpp` | `common_fit_check` / `--fit-check` (reuses the upstream no-alloc projection) | refuse plans that do not fit before allocating |
| `src/llama-model.cpp` | one log line per contiguous layer range with its device | placement visible in the log |
| `tools/rpc/test-rpc-small.cpp`, `tools/rpc/CMakeLists.txt` | `llama-rpc-test` (Gate 1) | small remote check before a model load |

Removed again (were Gemini's): CUDA event-wait change, RPC caps change, `LLAMAGLM_LAYER_SPLIT` / integer `-ts` override and topology banner, per-buffer VRAM guard, unified-memory hint in `llama-moe-cache.cpp`.

## Upstream behaviour the plan relies on

- `-ngl N` offloads the last N of `n_layer_all + 1` slots; `-ts` with integer counts summing to N gives exact counts per `--device` entry (`llama-model.cpp`, split computation before `get_layer_buft_list`).
- `--device` order is kept; RPC devices are named `RPC<n>` in the order of `--rpc` and of each server's `-d`.
- With mmap, CPU-resident weights stay in file-backed pages (`CPU_Mapped` buffers); mmap is turned off for the whole model if any device lacks `mmap_support`.
- The RPC client aborts the process if a worker connection fails (`RPC_STATUS_ASSERT`); a worker frees a disconnected client's buffers and waits for the next one.
- `rpc-server -c` caches weight tensors by hash in `LLAMA_CACHE`; later loads send only the hash. This is a load-time cache, not inference-time expert paging.
