# Current validation

| Capability | Status | Evidence/limit |
| --- | --- | --- |
| Source | PARTIAL | Planner, owned-layer evaluator, frame codec and mutual proof components; no distributed executor |
| Local CPU fixtures | CPU_FIXTURE_PASS | 4 CTest targets (8 planner cases, 12 wire/auth suites); 3 GLM tests including owned loading/forward; 2 non-GLM regression tests |
| Real tensor descriptors | PASS | 1,412 descriptors across all four pinned headers; no tensor data loaded |
| Windows compile | WINDOWS_COMPILE_PASS (CPU/SM89/SM120a) | 0e720734d, run 39520456801: 4 CTest targets (9 wire suites), 3 GLM CPU tests; newer mutual-proof changes await their own run |
| Three-process TCP | THREE_PROCESS_TCP_NOT_RUN | Transport not implemented |
| Ada GPU | SM89_GPU_NOT_RUN | No physical GPU tests |
| Blackwell GPU | SM120_GPU_NOT_RUN | No physical GPU tests |
| Three-PC full model | THREE_PC_FULL_MODEL_NOT_RUN | No weights or inference PCs used |
| 1M execution | ONE_MILLION_CONTEXT_NOT_RUN | Only integer/state-size counters exercised |

Local commands after a CPU CMake build:

```sh
ctest --test-dir build -R '^(test-glm-stage|test-glm-plan|test-glm-wire|test-gguf)$' --output-on-failure
build/bin/test-llama-archs --arch glm5-next --backend CPU --seed 53 -v 5
```

Require a nonzero completed-test summary from the architecture runner. A zero exit with muted/no summary is insufficient evidence by itself. Tests are assertions/checks in Release builds too; new checks do not rely on disabled C assertions.

Wire/auth tests require `-DLLAMA_GLM_RPC=ON`. Twelve component suites exercise an RFC 4231 HMAC vector, independent Python-derived key/header/MAC/challenge vectors, every frame/handshake envelope byte and selected payload corruptions, wrong identity/direction/channel/key, replay/skipped frame and handshake rejection, fatal invalidation, forged lengths/counts before payload allocation, control/data separation and a 128 MiB logical boundary processed in bounded 4 MiB fragments. Local backend: OpenSSL 3.5.7. The first nine frame suites also pass native Windows CNG at 0e720734d; the new three handshake suites await their own Windows run. These are component tests with no sockets, fragment reassembly, model execution or distributed transaction semantics.

Owned-loader qualification uses a synthetic six-layer GLM fixture with small dimensions. It verifies exact rank tensor sets, byte-for-byte provider/file payload equality, owned byte counters, invalid plans/modes, refusal of ordinary context creation/export, and closure of file handles. The source GGUF is removed after use. The experimental side header compiles as C and exports unmangled model/context symbols in the shared library.

Local forward qualification executes three independent owned models/contexts in one process against an unchanged monolithic six-layer oracle. Two splits (2/2/2 and 1/1/4) include pure-KDA and pure-DSA stages. The fixture selects eight of sixteen experts, retains the shared expert, and uses sparse index top-k 8. It processes 131 tokens in chunks 1/3/4/17/31/64/11, crossing four-token pool tails and a two-ubatch chunk. Both captured oracle boundaries are byte-exact; the largest final-logit absolute error with terminal last-row pruning is 3.35276127e-8 (seed 53, GCC 14.2 CPU). Tests inspect every layer's FP32 recurrent and FP16 MLA/index allocation and require absence of unowned state. A/B output all residual rows even when no row requests logits. Negative cases cover wrong contexts/cache precision, duplicate positions, B token input, encoder/noncausal modes and disabled headless output. Injected CPU aborts invalidate the context permanently; clearing memory cannot re-admit it, while recreating the context retains the loaded weights. This is not three-process TCP or GPU execution.

The same stage fixture passes native MSVC CPU in CPU and CUDA-linked Windows builds at 2a5360405 and 0e720734d. Downloaded 0e720734d CPU evidence reports exact boundaries and maximum logit absolute error 4.47034836e-8. Its LastTest.log reports nine wire suites; PE inspection confirms test-glm-wire.exe imports bcrypt.dll, with no OpenSSL DLL. The earlier 1410c4041 CUDA artifacts have a real PE delay-import directory and no direct nvcuda.dll import, and their CPU launch/tests passed. These establish native compilation and hosted CPU behavior, not GPU execution or standalone dependency packaging.

No target-model logits, full-context allocations, GPU residency, timing or tokens/sec measurements exist. Windows Actions will compile native application targets and run CPU fixtures; hosted runners do not prove NVIDIA inference correctness.

The planner integration cases construct tiny metadata-only GGUF files describing large tensors without allocating those payloads. They cover valid signed/unsigned schedules, middle-stage missing projections, incorrect expert counts, truncated headers, invalid splits, excessive counts and context refusal. Temporary paths include spaces to exercise native argument quoting on Windows. The fixtures are structural, not quant-codec or inference tests.
