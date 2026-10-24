# Decisions

## D1. Stock llama.cpp RPC, not a custom pipeline engine
Stage A runs `llama-server`; B and C run `ggml-rpc-server`. GLM-5.3-Flash support is upstream code (b11780). Sol's custom engine (fork branch `codex/glm53-3pc-rpc`) is not used. Keeping the delta to upstream small makes the official nvcc build of the same source a usable reference.

## D2. Use the RAM of all three PCs (six memory pools)
The model needs 109.5 GiB (text only). The three GPUs have 48 GB, so a GPU-only split is impossible (that is what produced the 55 GiB allocation). Each worker exposes `-d CUDA0,CPU`, so its RAM becomes a second RPC device. Stage A's CPU layers stay mmapped from its SSD. Exact plan: `MEMORY_PLAN.md`.

## D3. Exact layer counts with upstream `-ngl` / `-ts`
Integer `-ts` counts that sum to `-ngl` give exact per-device counts with the upstream split code. The launcher enforces the sum. Gemini's `LLAMAGLM_LAYER_SPLIT` override was removed: it changed the meaning of `-ts` silently and caused the "all GPU layers on the 4090" behaviour at `-ngl 15/20`.

## D4. Refuse plans that do not fit, before allocating
`--fit-check` (default on) and Gate 2 use the existing upstream no-allocation projection. Workers have their own reserve. A refused plan costs seconds; an overcommitted one froze Windows.

## D5. mmap on, no unified memory, no sysmem fallback, no op offload at first
mmap keeps Stage A's CPU weights in clean, droppable pages. `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is cleared by the launchers. The NVIDIA "Prefer No Sysmem Fallback" setting makes a full GPU fail fast. `--no-op-offload` avoids copying 1.3 GiB expert tensors into the 4090 during prompt processing; `-OpOffload` re-enables it. `--no-repack` keeps Stage A's CPU weights in the mmap (no 2.8 GiB anonymous copies of Q6_K/Q8_0).

## D6. Text only, 8K context, one slot
`-c 8192 -np 1`, MTP layer not loaded, no vision projector, `--cache-ram 0` (the prompt cache would compete with mmapped weights on Stage A), no `--context-shift` (not valid for the hybrid recurrent model).

## D7. Check cheaply first
Per-GPU CUDA self-test (`test-backend-ops` with the model's ops and weight types) -> remote device check (`llama-rpc-test`) -> memory projection -> load -> real tokens. Each step stops the run on failure.

## D8. Keep the clang cross build, but fix its math flags and keep an nvcc reference
The VM cannot run nvcc for Windows. clang's `-ffast-math` must not imply finite math for ggml (`-fno-finite-math-only`). The official b11780 build is the tie-breaker for toolchain questions.

## D9. Stay on b11780 for now
Newer upstream commits (to `a37cc4610`) bring no fix needed here; staying on b11780 keeps our build and the official reference identical in source.

## D10. Build the Windows package with MSVC + nvcc on GitHub Actions (v0.8.1)
The clang cross build of D8 is not an upstream-supported way to build ggml-cuda. GitHub's Windows runners build with nvcc, the toolchain of the official releases. CUDA version (12.8), GPU targets (sm_89, sm_120a), CPU level (AVX2) and launchers stay as in v0.8.0, so plans and gates do not change.
