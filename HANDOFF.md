# LlamaGLM engineering checkpoint

Development branch: `codex/glm53-3pc-rpc`. Current source is the HEAD of this branch; resolve with `git rev-parse HEAD`. Pinned upstream base: `1242be66f65d5d8a3d996ba56d6de106ced98cf6`.

This is an implementation project, not a deployable three-PC candidate. Work alone. Continue ordinary source changes, privacy-safe commits/pushes to this branch, and standard Windows GitHub Actions builds. Leave master and all other repositories unchanged. Do not create a worktree, new repository, PR or release. Do not download model tensor payloads to cloud or CI.

Implemented: native `llama-glm-plan`, exact three-rank contiguous ownership validation, bounded GGUF header parsing, full published tensor inventory and per-stage persistent-state arithmetic. Experimental `include/llama-stage.h` loads only the assigned GLM weights from providers/files, with byte parity and exact payload accounting tested for all ranks. It refuses monolithic context creation/model export and unsupported mmap/lazy/MTP modes. The metadata fetcher pins the Unsloth revision and downloads only the four exact header ranges. Existing GLM/MTP/SLRU implementations are inherited from the pinned base.

Current limits: owned weights can load, but no owned-layer execution/state filtering, stage RPC, desktop SSD provider, laptop residency enforcement, distributed API, rollback, prefix or snapshots yet. Existing `llama-server` remains the upstream monolithic server. Do not present it as the three-PC product. Default planned context is 1,048,576, wire FP32, MLA/index FP16, KDA state FP32 as in upstream. Preserve all eight selected experts and shared experts.

CPU verification: three CTest targets include eight native planner integration cases. `test-llama-archs --arch glm5-next --backend CPU --seed 53 -v 5` now reports two completed tests (owned-weight loading qualification plus inherited monolithic inference/roundtrip/mixed batches). A non-GLM `--arch llama` regression reports two passes. Zero-test runs now fail; queued logs flush before exit. No GPU or full-model inference ran.

Windows build: workflow `.github/workflows/glm-windows.yml` builds native CPU and CUDA 12.8.1 application targets (SM89 and SM120a), then CPU fixtures. Best Windows CPU-tested SHA is `acae9b6aa05bfaed97f10f50be2bac2f223530d6`, run `39512089611`, artifact `12305654727`. Its CUDA compile/link passed, then test-gguf failed process load (`0xc0000135`): PE inspection found a direct nvcuda.dll import on the CPU runner. Supported MSVC driver delay loading plus explicit verified runtime paths is the current fix awaiting CI. VMM remains enabled. First run `39511902026` had a Windows shared-library fixture-target failure, fixed by static libraries. Native validation of the newer planner/owned-loader source remains pending; old Windows passes do not cover it.

Next actions:

1. Push this reviewed checkpoint; inspect the exact Windows run and fix toolchain/build/test failures.
2. Extend the now-tested metadata parser/FFN projection inventory to a complete source-derived attention/KDA tensor schema. Eight native-planner integration cases and expert slab-bound tests are present locally; build them on Windows after pushing this next checkpoint.
3. Extend the experimental owned-weight interface with actual stage execution. Preserve the upstream monolithic graph as an oracle. Filter KDA/DSA/index state by the same validated range and pass all four FP32 mHC streams across boundaries. Graph input widths, scheduler/output widths, headless outputs, terminal head and dummy reserve batches need explicit handling. `llama_init_from_model` deliberately refuses partial models until a dedicated stage path exists. Never emulate ownership by computing unowned layers elsewhere.
4. Implement bounded native transport and state transactions, then desktop SSD tiers and fail-closed resident laptop provider. Read `docs/ARCHITECTURE.md`, `docs/RPC_PROTOCOL.md`, `docs/MEMORY_PLAN.md` and `docs/OPEN_ISSUES.md` first.

Research and build trees outside the checkout are regenerable and are not the durable source of truth. Keep meaningful code and evidence committed and pushed. Qwen continuation is optional and must only be prepared if explicitly requested; it would use an offline OpenCode workspace and SSH-only Linux CPU build server, with no authenticated GitHub/Actions access.
