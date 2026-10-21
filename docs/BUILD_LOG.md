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

## 2026-10-20 - owned-weight loading (local, next checkpoint)

Added experimental provider/file stage loaders without changing the stable model/context parameter structs. All three ranks match independent tensor-set expectations and the complete synthetic model's bytes; exact owned payload counters pass. Partial models refuse monolithic evaluation/export, mmap, lazy model reads and unsupported MTP ownership. The GLM runner now completes two tests; non-GLM llama completes two regression tests. C header compilation and exported C symbols verified. Runner logs are flushed before exit (the singleton logger otherwise drops queued summaries); zero-test runs now fail and the negative case is verified.

Diagnosed local integration fixes: this base's ordinary buffered mode is named NONE, not READ; lazy defaults to AUTO and must be explicitly OFF for staged loading. Progress size is computed in init_mappings, so exclusion now removes unowned entries from the weight lookup before that pass rather than subtracting from the not-yet-populated byte total. The source file/data buffers remain owned by the normal loader lifecycle.

Windows run `39513276396` for `61a5594428a9d20935668fd14943b1777842c69e` was cancelled while pending because it is superseded by the owned-loader checkpoint. No Windows pass is claimed for that SHA. The acae9b6aa CUDA run remains active. The next workflow adds a pinned, checksum-verifying ccache action (4.13.6 Windows binary) and a 2G per-job compiler cache to retain valid compilation outputs for subsequent cycles.

## 2026-10-20 - bound the CUDA build loop

Owned-loader source checkpoint: `d7adf7bcab305f8c0a887e8a88e9b222a61177fc`; pushed and independently confirmed through read-only Git. Native Windows run `39515432129` was superseded by a workflow that separates SM89 and SM120 compilation and explicitly builds the required FP16/BF16 attention-cache variants. All quantized weight codecs remain in the backend. Two compiler jobs per runner and 2G fixed-key verified ccache per matrix entry bound resources. Both required architectures will be rebuilt at the current source SHA.

The older combined CUDA job completed before cancellation: all 380 compile/link steps passed for both SM89 and SM120a, including real application executables. The job then failed `test-gguf` at Windows process load with `0xc0000135`; GLM inference tests were consequently unrun in that job. Downloaded [CUDA artifact 12307694188](https://github.com/Protonburn35/LlamaGLM/actions/runs/39512089611/artifacts/12307694188) (165,066,811 bytes), inspected PE imports using objdump, and confirmed a direct `nvcuda.dll` import. Hosted CPU runners do not supply that NVIDIA driver DLL. Fix: use supported MSVC delay loading of the optional driver; existing CUDA initialization checks cudaGetDeviceCount before driver API use. Also discover directories of the verified cudart/cuBLAS runtime DLLs explicitly. VMM and model operators remain enabled; no fake driver or verification bypass is introduced. Retain dependency reports in future artifacts and rerun the actual CPU checks in CUDA-linked builds.

## 2026-10-21 - local owned-layer forward evaluation

Extended the experimental stage API with full-context defaults and dedicated context creation. Graphs, weights and KDA/MLA/index state share the validated ownership range. B/C batch widths and A/B output widths carry all four FP32 streams; only C evaluates the head. The append-only baseline refuses invalid precision, contexts, positions and modes before state preparation, and refuses all reuse after preparation/compute failure. Context recreation preserves loaded weights.

GCC 14.2 CPU: three CTest targets passed (eight planner cases), GLM runner completed three tests, non-GLM llama two tests. Six-layer forward fixtures use splits 2/2/2 and 1/1/4, eight selected of sixteen experts plus shared expert, sparse index top-k 8, and 131 tokens chunked 1/3/4/17/31/64/11 with ubatch 32. Both boundary captures match exactly; largest final-logit absolute error is 3.35276127e-8 with terminal row pruning. State allocation ownership/precision, invalid input/settings, CPU abort invalidation and context recreation pass. C header syntax and unmangled stage context exports verified. No GPU, TCP, full quant or 1M inference is implied.

Issues found by these tests: headless graphs must not create unused output-ID inputs; stages with only KDA or only DSA must omit the other memory inputs, including pool inputs. Otherwise input population dereferences buffers that the scheduler correctly did not allocate. Fixed graph construction rather than suppressing missing-buffer assertions. A recovery fixture initially reused a batch bound to a destroyed context; it now constructs a new batch for the new context.

[Run 39516895221](https://github.com/Protonburn35/LlamaGLM/actions/runs/39516895221), source `1410c4041c29fdbc9a875288f9150d150e62c6cd`: Windows CPU job `118951103807` passed with MSVC 19.44.35229.0, three CTest targets, two then-existing GLM tests, quant metadata inventory and application launch checks. [CPU artifact 12308069497](https://github.com/Protonburn35/LlamaGLM/actions/runs/39516895221/artifacts/12308069497), 11,272,043 bytes. CUDA jobs `118951103327` (SM89) and `118951103402` (SM120a) remain in compile at this entry. This run does not contain the new forward evaluator; its own Windows source-specific run is pending.

Stage-forward checkpoint `2a536040552a23c76fa31cc849e000a0d114e281` was pushed. [Run 39519238289](https://github.com/Protonburn35/LlamaGLM/actions/runs/39519238289) is pending behind the active CUDA jobs; no Windows stage-forward pass is claimed yet.

## 2026-10-21 - authenticated frame components

Added optional `LLAMA_GLM_RPC` C++17 frame/crypto components. Windows uses native BCrypt; Linux CPU fixtures use OpenSSL 3.5.7. Explicit LE fixed headers, checked FP32 four-stream dimensions/positions, bounded fragment/control sizes, domain-separated per-direction/channel/identity/nonce keys, full header+payload HMAC, strict frame sequence and permanent receiving-codec refusal after errors are implemented. No sockets, handshake driver, fragment assembler or transaction semantics are present.

Local native test-glm-wire passes nine suites, including RFC 4231 and independent Python struct/HMAC vectors, mutation of every header byte, payload corruption, link binding changes, replay/skipped sequence and forged lengths before allocation. A 128 MiB logical boundary is exercised with one retained 4 MiB payload/packet at a time, without model data. All four focused CTest targets pass. CI now explicitly enables the component and builds/runs its real CNG-backed native test. Windows CNG results await that exact future source run.
