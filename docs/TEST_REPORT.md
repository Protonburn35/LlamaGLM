# Current validation

| Capability | Status | Evidence/limit |
| --- | --- | --- |
| Source | PARTIAL | Planner, owned-weight loaders and local owned-layer evaluator; no distributed executor |
| Local CPU fixtures | CPU_FIXTURE_PASS | 4 CTest targets (8 planner cases, 9 wire suites); 3 GLM tests including owned loading/forward; 2 non-GLM regression tests |
| Real tensor descriptors | PASS | 1,412 descriptors across all four pinned headers; no tensor data loaded |
| Windows compile | WINDOWS_COMPILE_PASS (CPU); CUDA pending | 1410c4041, run 39516895221 CPU passed; stage-forward changes await their own Windows build |
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

Wire tests require `-DLLAMA_GLM_RPC=ON`. Nine component suites exercise an RFC 4231 HMAC vector, independent Python-derived key/header/MAC vectors, every header byte and selected payload corruptions, wrong link identity/direction/channel, replay/skipped frame rejection, fatal receiver invalidation, forged lengths/counts before payload allocation, control/data separation and a 128 MiB logical boundary processed in bounded 4 MiB fragments. Local crypto backend: OpenSSL 3.5.7. Native Windows CNG qualification is pending CI. These are byte-codec tests with no sockets, handshake, fragment reassembly, model execution or transaction semantics.

Owned-loader qualification uses a synthetic six-layer GLM fixture with small dimensions. It verifies exact rank tensor sets, byte-for-byte provider/file payload equality, owned byte counters, invalid plans/modes, refusal of ordinary context creation/export, and closure of file handles. The source GGUF is removed after use. The experimental side header compiles as C and exports unmangled model/context symbols in the shared library.

Local forward qualification executes three independent owned models/contexts in one process against an unchanged monolithic six-layer oracle. Two splits (2/2/2 and 1/1/4) include pure-KDA and pure-DSA stages. The fixture selects eight of sixteen experts, retains the shared expert, and uses sparse index top-k 8. It processes 131 tokens in chunks 1/3/4/17/31/64/11, crossing four-token pool tails and a two-ubatch chunk. Both captured oracle boundaries are byte-exact; the largest final-logit absolute error with terminal last-row pruning is 3.35276127e-8 (seed 53, GCC 14.2 CPU). Tests inspect every layer's FP32 recurrent and FP16 MLA/index allocation and require absence of unowned state. A/B output all residual rows even when no row requests logits. Negative cases cover wrong contexts/cache precision, duplicate positions, B token input, encoder/noncausal modes and disabled headless output. Injected CPU aborts invalidate the context permanently; clearing memory cannot re-admit it, while recreating the context retains the loaded weights. This is not three-process TCP or GPU execution.

No target-model logits, full-context allocations, GPU residency, timing or tokens/sec measurements exist. Windows Actions will compile native application targets and run CPU fixtures; hosted runners do not prove NVIDIA inference correctness.

The planner integration cases construct tiny metadata-only GGUF files describing large tensors without allocating those payloads. They cover valid signed/unsigned schedules, middle-stage missing projections, incorrect expert counts, truncated headers, invalid splits, excessive counts and context refusal. Temporary paths include spaces to exercise native argument quoting on Windows. The fixtures are structural, not quant-codec or inference tests.
