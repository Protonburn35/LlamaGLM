# Current validation

| Capability | Status | Evidence/limit |
| --- | --- | --- |
| Source | PARTIAL | Planner, owned-layer evaluator, authenticated TCP and test-only three-process workers; no production coordinator |
| Local CPU fixtures | CPU_FIXTURE_PASS | 6 CTest targets (8 planner cases, 12 wire/auth and 8 TCP suites, three-process/missing-middle cases); 3 GLM tests; 2 non-GLM tests |
| Real tensor descriptors | PASS | 1,412 descriptors across all four pinned headers; no tensor data loaded |
| Windows compile | WINDOWS_COMPILE_PASS (CPU/SM89/SM120a) | e5a94c203, run 39526159488: 6 CTest targets including three-process inference, 3 GLM CPU tests |
| Three-process TCP | THREE_PROCESS_CPU_FIXTURE_PASS | Actual A/B/C CPU processes, six synthetic layers, 131 tokens; GCC/MSVC boundaries/logits byte-exact |
| Ada GPU | SM89_GPU_NOT_RUN | No physical GPU tests |
| Blackwell GPU | SM120_GPU_NOT_RUN | No physical GPU tests |
| Three-PC full model | THREE_PC_FULL_MODEL_NOT_RUN | No weights or inference PCs used |
| 1M execution | ONE_MILLION_CONTEXT_NOT_RUN | Only integer/state-size counters exercised |

Local commands after a CPU CMake build:

```sh
ctest --test-dir build -R '^(test-glm-stage|test-glm-plan|test-glm-wire|test-glm-net|test-glm-pipeline|test-gguf)$' --output-on-failure
build/bin/test-llama-archs --arch glm5-next --backend CPU --seed 53 -v 5
```

Require a nonzero completed-test summary from the architecture runner. A zero exit with muted/no summary is insufficient evidence by itself. Tests are assertions/checks in Release builds too; new checks do not rely on disabled C assertions.

Wire/auth tests require `-DLLAMA_GLM_RPC=ON`. Twelve component suites exercise an RFC 4231 HMAC vector, independent Python-derived key/header/MAC/challenge vectors, every frame/handshake envelope byte and selected payload corruptions, wrong identity/direction/channel/key, replay/skipped frame and handshake rejection, fatal invalidation, forged lengths/counts before payload allocation, control/data separation and a 128 MiB logical boundary processed in bounded 4 MiB fragments. Local backend: OpenSSL 3.5.7; all twelve suites also pass native Windows CNG at 79d638759. These are byte-component tests without fragment reassembly, model execution or distributed transaction semantics.

Eight additional TCP suites pass on local Linux and native Windows Winsock at 52d499a2e with actual loopback sockets in one process: fresh mutual proof, bit-preserving 4 MiB/64 KiB two-way fragments, cancel acknowledgement on an independent control socket while data is backpressured, deadline/EOF permanent shutdown, wrong secret refusal, malformed lengths rejected without reading a body, explicit interface/port conflict and concurrent read/listener interruption. No model executes in these component tests.

The separate test-glm-pipeline launches three distinct native CPU worker processes. Each loads only its own layers through the stage loader (2/2/2), owns its own recurrent/MLA/index state, and computes actual graph work. A sends all four FP32 streams to B, B computes and forwards to C, and C runs the final head. A separate monolithic generator writes both reference boundaries and all logits before workers start. For 131 tokens over seven uneven chunks, both boundaries and logits are byte-exact under GCC 14.2 and MSVC 19.44 (maximum error 0). The independent tensor-byte total equals the sum of the three reported stage payloads. Direct A-to-C startup without B refuses before any inference output. Windows CPU and both CUDA-linked builds pass at e5a94c203; all model execution remains CPU.

The worker binary is test-only, loopback-only, requires a fresh per-run secret file from the harness, and refuses models without the small six-layer fixture marker/geometry. No worker credential default is embedded or logged. It creates no user service. The generated 80 MiB F32 GGUF and references are temporary synthetic data, not downloaded real weights. Context is 256, not 1M. Production identity/credential provisioning, multi-fragment execution, HTTP/sampling, session transactions, cancellation/rollback, residency and SSD tiers remain absent. These results must not be presented as physical three-PC/full-quant/GPU validation.

Owned-loader qualification uses a synthetic six-layer GLM fixture with small dimensions. It verifies exact rank tensor sets, byte-for-byte provider/file payload equality, owned byte counters, invalid plans/modes, refusal of ordinary context creation/export, and closure of file handles. The source GGUF is removed after use. The experimental side header compiles as C and exports unmangled model/context symbols in the shared library.

Local forward qualification executes three independent owned models/contexts in one process against an unchanged monolithic six-layer oracle. Two splits (2/2/2 and 1/1/4) include pure-KDA and pure-DSA stages. The fixture selects eight of sixteen experts, retains the shared expert, and uses sparse index top-k 8. It processes 131 tokens in chunks 1/3/4/17/31/64/11, crossing four-token pool tails and a two-ubatch chunk. Both captured oracle boundaries are byte-exact; the largest final-logit absolute error with terminal last-row pruning is 3.35276127e-8 (seed 53, GCC 14.2 CPU). Tests inspect every layer's FP32 recurrent and FP16 MLA/index allocation and require absence of unowned state. A/B output all residual rows even when no row requests logits. Negative cases cover wrong contexts/cache precision, duplicate positions, B token input, encoder/noncausal modes and disabled headless output. Injected CPU aborts invalidate the context permanently; clearing memory cannot re-admit it, while recreating the context retains the loaded weights. This is not three-process TCP or GPU execution.

The same stage fixture passes native MSVC CPU in CPU and CUDA-linked Windows builds at 2a5360405 and 0e720734d. Downloaded 0e720734d CPU evidence reports exact boundaries and maximum logit absolute error 4.47034836e-8. Its LastTest.log reports nine wire suites; PE inspection confirms test-glm-wire.exe imports bcrypt.dll, with no OpenSSL DLL. The earlier 1410c4041 CUDA artifacts have a real PE delay-import directory and no direct nvcuda.dll import, and their CPU launch/tests passed. These establish native compilation and hosted CPU behavior, not GPU execution or standalone dependency packaging.

No target-model logits, full-context allocations, GPU residency, timing or tokens/sec measurements exist. Windows Actions will compile native application targets and run CPU fixtures; hosted runners do not prove NVIDIA inference correctness.

The planner integration cases construct tiny metadata-only GGUF files describing large tensors without allocating those payloads. They cover valid signed/unsigned schedules, middle-stage missing projections, incorrect expert counts, truncated headers, invalid splits, excessive counts and context refusal. Temporary paths include spaces to exercise native argument quoting on Windows. The fixtures are structural, not quant-codec or inference tests.
