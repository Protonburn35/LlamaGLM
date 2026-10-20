# LlamaGLM engineering checkpoint

Development branch: `codex/glm53-3pc-rpc`. Current source is the HEAD of this branch; resolve with `git rev-parse HEAD`. Pinned upstream base: `1242be66f65d5d8a3d996ba56d6de106ced98cf6`.

This is an implementation project, not a deployable three-PC candidate. Work alone. Continue ordinary source changes, privacy-safe commits/pushes to this branch, and standard Windows GitHub Actions builds. Leave master and all other repositories unchanged. Do not create a worktree, new repository, PR or release. Do not download model tensor payloads to cloud or CI.

Implemented: native `llama-glm-plan`, exact three-rank contiguous ownership validation, bounded GGUF header parsing, full published tensor inventory and per-stage persistent-state arithmetic. The metadata fetcher pins the Unsloth revision and downloads only the four exact header ranges. Existing GLM/MTP/SLRU implementations are inherited from the pinned base.

Current limits: no owned-layer execution, stage RPC, desktop SSD provider, laptop residency enforcement, distributed API, rollback, prefix or snapshots yet. Existing `llama-server` remains the upstream monolithic server. Do not present it as the three-PC product. Default planned context is 1,048,576, wire FP32, MLA/index FP16, KDA state FP32 as in upstream. Preserve all eight selected experts and shared experts.

CPU verification: `test-glm-stage`, `test-gguf` and the existing `test-llama-archs --arch glm5-next --backend CPU --seed 53 -v 5` pass on the cloud machine. The latter reports one completed architecture test, with roundtrip and mixed/chunked-batch checks. No GPU or full-model inference ran.

Windows build: workflow `.github/workflows/glm-windows.yml` builds native CPU and CUDA 12.8.1 application targets (SM89 and SM120a), then CPU fixtures. First run `39511902026` at `831afe27b1ce4d43eb1b3e25f9c99befb98602d5` failed before compiling: Windows shared libraries disable the internal architecture fixture target. Static-library configuration is the diagnosed fix; rebuild pending. No passing Windows SHA/artifact exists yet. Update `docs/BUILD_LOG.md` and this file after observing actual runs.

Next actions:

1. Push this reviewed checkpoint; inspect the exact Windows run and fix toolchain/build/test failures.
2. Add independent metadata-parser and quant-inventory regression tests, including malformed headers and complete projection ownership checks.
3. Introduce an experimental owned-layer graph/loader interface, preserving the upstream monolithic graph as an oracle. Filter weights and KDA/DSA/index state by the same validated range; pass all four FP32 mHC streams across boundaries. Graph inputs, scheduler outputs and terminal head need explicit handling. Never emulate ownership by computing unowned layers elsewhere.
4. Implement bounded native transport and state transactions, then desktop SSD tiers and fail-closed resident laptop provider. Read `docs/ARCHITECTURE.md`, `docs/RPC_PROTOCOL.md`, `docs/MEMORY_PLAN.md` and `docs/OPEN_ISSUES.md` first.

Research and build trees outside the checkout are regenerable and are not the durable source of truth. Keep meaningful code and evidence committed and pushed. Qwen continuation is optional and must only be prepared if explicitly requested; it would use an offline OpenCode workspace and SSH-only Linux CPU build server, with no authenticated GitHub/Actions access.
