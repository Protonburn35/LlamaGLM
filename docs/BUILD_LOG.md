# Build and test ledger

## 2026-10-20 - first planning component (pre-push working tree)

Base: `1242be66f65d5d8a3d996ba56d6de106ced98cf6`. Branch: `codex/glm53-3pc-rpc`. Cloud Linux x86_64, GCC 14.2, CMake 3.31.6, Ninja 1.11.1.4, CPU AVX2, four compile jobs. Native planner and inherited llama library compiled. Two CTest targets (`test-glm-stage`, `test-gguf`) passed; GLM architecture fixture reported one completed passing test with seed 53 and verbosity 5. Quant-header fetch and all 1,412 descriptor inventory succeeded. These local results precede the checkpoint commit and are not Windows results.

Setup fixes during development: the new test was initially gated on a target created later in CMake; switched to the actual tools-build condition. The publisher stores the KV-head array as INT32 and includes NextN in block_count; parser now validates both accepted integer array types and subtracts the explicit NextN count before target ownership. NVIDIA's referenced 13.4.0 redistributable manifest returned 404; selected the accessible official CUDA 12.8.1 manifest, whose compiler supports SM120. Pinned NVIDIA component SHA256 values and a specific CCCL commit; no verification bypass.

Actions administration-policy endpoint returned 403 for the integration; ordinary repository metadata, Git read, branch read and Actions run listing worked. Actual Windows workflow execution remains pending the first push; the administration endpoint denial alone does not establish Actions are unavailable.

Windows run URL/ID, source SHA, failed step, root cause and fix SHA will be appended after actual execution. No Windows artifacts have been published at this checkpoint. GPU, three-process TCP, three-PC model and 1M runtime tests have not run.
