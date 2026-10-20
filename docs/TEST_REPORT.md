# Current validation

| Capability | Status | Evidence/limit |
| --- | --- | --- |
| Source | PARTIAL | Planner and owned-weight loaders; no stage evaluator or distributed executor |
| Local CPU fixtures | CPU_FIXTURE_PASS | 3 CTest targets (8 planner cases); 2 GLM tests including owned weights; 2 non-GLM regression tests |
| Real tensor descriptors | PASS | 1,412 descriptors across all four pinned headers; no tensor data loaded |
| Windows compile | WINDOWS_COMPILE_PASS (CPU); CUDA pending | acae9b6aa, run 39512089611; newer local planner changes await Windows build |
| Three-process TCP | THREE_PROCESS_TCP_NOT_RUN | Transport not implemented |
| Ada GPU | SM89_GPU_NOT_RUN | No physical GPU tests |
| Blackwell GPU | SM120_GPU_NOT_RUN | No physical GPU tests |
| Three-PC full model | THREE_PC_FULL_MODEL_NOT_RUN | No weights or inference PCs used |
| 1M execution | ONE_MILLION_CONTEXT_NOT_RUN | Only integer/state-size counters exercised |

Local commands after a CPU CMake build:

```sh
ctest --test-dir build -R '^(test-glm-stage|test-glm-plan|test-gguf)$' --output-on-failure
build/bin/test-llama-archs --arch glm5-next --backend CPU --seed 53 -v 5
```

Require a nonzero completed-test summary from the architecture runner. A zero exit with muted/no summary is insufficient evidence by itself. Tests are assertions/checks in Release builds too; new checks do not rely on disabled C assertions.

Owned-loader qualification uses a synthetic six-layer GLM fixture with small dimensions. It verifies exact rank tensor sets, byte-for-byte provider/file payload equality, owned byte counters, invalid plans/modes, refusal of ordinary context creation/export, and closure of file handles. The source GGUF is removed after use. This is three model-loading paths, not three-process TCP or distributed layer computation. The new side header also compiles as C and exports unmangled symbols in the shared library.

No target-model logits, full-context allocations, GPU residency, timing or tokens/sec measurements exist. Windows Actions will compile native application targets and run CPU fixtures; hosted runners do not prove NVIDIA inference correctness.

The planner integration cases construct tiny metadata-only GGUF files describing large tensors without allocating those payloads. They cover valid signed/unsigned schedules, middle-stage missing projections, incorrect expert counts, truncated headers, invalid splits, excessive counts and context refusal. Temporary paths include spaces to exercise native argument quoting on Windows. The fixtures are structural, not quant-codec or inference tests.
