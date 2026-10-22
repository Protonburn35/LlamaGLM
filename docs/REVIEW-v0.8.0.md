# v0.8.0 import review

The user-provided VM source contains an upstream llama.cpp b11780 engine plus repaired RPC diagnostics, memory admission, Windows launchers, and cross-build support. The 13 changed engine files identified in the supplied upstream audit were compared with GitHub `master` at `1242be66f65d5d8a3d996ba56d6de106ced98cf6`. Downloaded RPC transport files and the root CMake configuration matched that baseline. The original snapshot labels are engine `502ff59` and launchers `6664b7d`; their unpublished Git history was not imported.

Claude's repairs remove the need for Gemini's custom layer-split and event-wait workarounds, restore upstream RPC mmap capabilities, and preserve infinity comparisons in Clang CUDA kernels. These are sensible corrections. The local checks below support the repaired CPU/RPC behavior; they do not establish the cause of the earlier hardware crash.

## Corrections made during this review

- A missing CUDA test executable, absent CUDA backend, or zero-test run could report a successful self-test. The launcher now requires a successful CUDA0 result and a nonzero executed-test count for each invocation. Explicit skip switches remain available.
- The worker reserve check added a remote allocation size to the reserve before comparing, allowing integer overflow. It now checks the reserve first and subtracts before comparing the request. Invalid, negative, overflowing, empty, trailing-comma, and excess-device settings refuse allocations.
- The exported packaging script assumed separate nested Git repositories. It now packages the single GitHub checkout, with an optional `LLAMAGLM_BUILD_DIR` build location.
- The custom Clang placement-new shim and emulated TLS are restricted to MinGW and use the checked-in shim. Ordinary Clang CUDA builds no longer require an external copy of that header.
- Log collection no longer promises that collected server/system logs contain no private data.

RPC intentionally remains unauthenticated. The documented 8K context, stock-RPC scheduling, and memory placement are accepted prototype behavior. The custom stage-runtime work is retained on its separate branch.

## Independently repeated checks

Linux cloud workspace, GCC 14.2, CMake 3.31.6, Ninja 1.11.1.4, CPU backend, no GPU:

- All selected binaries built: server, completion, fit tool, RPC worker and gate tool, backend-op test, architecture test, GGUF test, argument parser test, and multi-server regression.
- CTest: 3/3 passed (`test-arg-parser`, `test-gguf`, `test-rpc-multi-server`).
- Upstream GLM5-next CPU architecture fixture: 1/1 passed, including serialization and mixed execution checks.
- Actual RPC gate: 96 graph comparisons passed over two independent workers with two CPU devices each, covering MUL_MAT/MUL_MAT_ID, six weight types, and one/eight-row shapes. A deliberately implausible device-memory range failed as expected.
- A generated random-weight GLM fixture produced identical generation bytes locally and over two RPC workers for 16 decode steps. Both workers recorded graph execution. This uses a tiny synthetic model, not the published full model.
- The fit tool accepted a small valid plan and refused a deliberately oversized margin before loading weights.
- `tests/test-rpc-reserve.py`: eight settings exercised actual wire requests, including UINT64_MAX allocation requests, allocation refusal for malformed settings, and continued service after refusal.
- `tests/test-llamaglm-launchers.ps1`: missing executable, actual CPU-only backend binary, absent CUDA, zero tests, failed tests, wrong backend, and successful ANSI-colored CUDA output. All expected outcomes passed under PowerShell 7.6.8. Mock CUDA output tests the launcher decision only.
- PowerShell parsing, Bash syntax, Python compilation, and Git whitespace checks passed.

The native Windows workflow compiles CPU, SM89, and SM120 variants and runs the CPU/worker tests plus the launcher regression under Windows PowerShell 5.1. Its result is recorded separately after the source push. It does not execute CUDA kernels.

The original VM build and Wine reports are preserved in BUILD_LOG.md and TEST_REPORT.md as imported evidence. No new full model tensors were downloaded, and no inference on the user's PCs was performed during this review. Performance numbers in LIMITATIONS.md remain estimates.

## GitHub Actions scope

The first pushes to `master` triggered dozens of inherited upstream workflows, including Android, Apple, unrelated accelerator backends, and self-hosted runners. Unrelated runs from this import were cancelled. Workflow configuration changes are deferred; the inherited triggers are still present. See [CI_SCOPE.md](CI_SCOPE.md) before pushing further changes. Keep AMD/Vulkan compatibility in the future CI plan.

The first import's Python lint run rejected a `print()` in the new reserve regression script. It now uses the repository's required logging convention; the reserve scenarios still execute normally.
