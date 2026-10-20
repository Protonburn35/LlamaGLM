# Build and test ledger

## 2026-10-20 - first planning component (pre-push working tree)

Base: `1242be66f65d5d8a3d996ba56d6de106ced98cf6`. Branch: `codex/glm53-3pc-rpc`. Cloud Linux x86_64, GCC 14.2, CMake 3.31.6, Ninja 1.11.1.4, CPU AVX2, four compile jobs. Native planner and inherited llama library compiled. Two CTest targets (`test-glm-stage`, `test-gguf`) passed; GLM architecture fixture reported one completed passing test with seed 53 and verbosity 5. Quant-header fetch and all 1,412 descriptor inventory succeeded. These local results precede the checkpoint commit and are not Windows results.

Setup fixes during development: the new test was initially gated on a target created later in CMake; switched to the actual tools-build condition. The publisher stores the KV-head array as INT32 and includes NextN in block_count; parser now validates both accepted integer array types and subtracts the explicit NextN count before target ownership. NVIDIA's referenced 13.4.0 redistributable manifest returned 404; selected the accessible official CUDA 12.8.1 manifest, whose compiler supports SM120. Pinned NVIDIA component SHA256 values and a specific CCCL commit; no verification bypass.

Actions administration-policy endpoint returned 403 for the integration; ordinary repository metadata, Git read, branch read and Actions run listing worked. Actual Windows workflow execution remains pending the first push; the administration endpoint denial alone does not establish Actions are unavailable.

Windows run URL/ID, source SHA, failed step, root cause and fix SHA will be appended after actual execution. No Windows artifacts have been published at this checkpoint. GPU, three-process TCP, three-PC model and 1M runtime tests have not run.

## 2026-10-20 - Windows candidate 831afe27b

Source: `831afe27b1ce4d43eb1b3e25f9c99befb98602d5`. [Run 39511902026](https://github.com/Protonburn35/LlamaGLM/actions/runs/39511902026), Windows Server 2022, MSVC, CPU/CUDA matrix. CPU configure succeeded; compile step failed before compiling because `test-llama-archs` is deliberately not created for Windows shared-library builds (it calls internal symbols). Fix: configure static llama libraries, preserving the real fixture and application targets. The CUDA job was still installing when this failure was diagnosed; cancel the superseded run and rebuild with the supported static configuration. The CPU artifact contains configuration evidence only, not passing binaries.

An unrelated inherited `build-cann.yml` also emitted a no-job failure ([39511897270](https://github.com/Protonburn35/LlamaGLM/actions/runs/39511897270)); its jobs are commented out in the pinned upstream file. It is not evidence about the GLM Windows build. No upstream CANN workflow is changed for this project.

## 2026-10-20 - Windows retry acae9b6aa

Source/fix SHA: `acae9b6aa05bfaed97f10f50be2bac2f223530d6`. [Run 39512089611](https://github.com/Protonburn35/LlamaGLM/actions/runs/39512089611). Windows x64 CPU job `118935341799` passed using MSVC 19.44.35229.0 and static llama libraries. Compiled native planner, llama-server, llama-cli and fixtures. Two CTest targets passed and the GLM architecture fixture completed one passing test; exact pinned quant metadata fetch/inventory and executable version launch checks passed. [CPU artifact 12305654727](https://github.com/Protonburn35/LlamaGLM/actions/runs/39512089611/artifacts/12305654727), name `glm-windows-cpu-acae9b6aa05bfaed97f10f50be2bac2f223530d6`, size 11,233,702 bytes, retention three days. CUDA job `118935342013` was still compiling when this entry was written. No GPU execution is implied.

Subsequent local source extends expert slab bounds, FFN inventory validation, optional MTP embedding dependency, and eight metadata-parser integration cases. All three local CTest targets pass; the original GLM fixture still reports one passing test. Six contiguous split comparisons generated from real headers. These changes are not covered by the acae9b6aa Windows pass until a new run builds them.
