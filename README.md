# LlamaGLM

GLM-5.3-Flash (`unsloth/GLM-5.3-Flash-GGUF`, `UD-IQ3_XXS`, 112.1 GiB) across three Windows 11 PCs on a LAN, using upstream llama.cpp (b11780) with `ggml-cuda` and `ggml-rpc` plus a small set of safety patches and launchers.

**Status: generates text on the three PCs, slowly.** Version 0.8.0 fixed the defects found in the 0.7.0 release (see `HANDOFF.md`). Version 0.8.1 is the same code plus the import-review fixes, built with MSVC and nvcc by a GitHub Actions workflow instead of the VM's clang cross-build. v0.8.1 ran on the three PCs on 2026-10-24 (1.71 tok/s generation, 3.42 tok/s prompt processing; [hardware handoff](docs/HARDWARE_HANDOFF-2026-10-24.md)). Version 0.8.2 adds optional speed experiments, all off by default ([test plan](docs/TEST-PLAN-v0.8.2.md)).

**Speed (measured 2026-10-24):** 1.71 tokens/s output and 3.42 tokens/s prompt processing. Likely cause and options: [`docs/SPEED_RESEARCH-2026-10-24.md`](docs/SPEED_RESEARCH-2026-10-24.md); older estimates in [`LIMITATIONS.md`](LIMITATIONS.md).

| PC | GPU / RAM | Role | Holds (default plan) |
|---|---|---|---|
| A 192.168.50.206 | RTX 4090 24 GB / 32 GB | `llama-server` API on `127.0.0.1:8081`, model files | layers 0-17 in RAM (mmap), 18-24 on the 4090 |
| B 192.168.50.129:50052 | RTX 4060 Laptop 8 GB / 32 GB | RPC worker (GPU + RAM) | 25-31 in RAM, 32-33 on the GPU |
| C 192.168.50.143:50053 | RTX 5070 Ti 16 GB / 32 GB | RPC worker (GPU + RAM) | 34-40 in RAM, 41-44 + output on the GPU |

The model does not fit in the 48 GB of the three GPUs, so each worker also offers its RAM, and Stage A pages part of its layers from its SSD. All weights keep their original quantization; all routed experts are computed; activations cross PCs in F32.

- First run: [`FIRST-RUN.md`](FIRST-RUN.md)
- Install details: [`INSTALL-WINDOWS.md`](INSTALL-WINDOWS.md)
- Status, review of 0.7.0, next steps: [`HANDOFF.md`](HANDOFF.md)
- Memory plan with exact sizes: [`docs/MEMORY_PLAN.md`](docs/MEMORY_PLAN.md)
- Limitations, speed estimates, hardware vs. implementation: [`LIMITATIONS.md`](LIMITATIONS.md)
- Open issues: [`docs/OPEN_ISSUES.md`](docs/OPEN_ISSUES.md)

Not supported yet: MTP/NextN speculation, vision, context beyond the 8K test setting.

The GitHub checkout contains the engine at its root and the launchers in `scripts/`. After building, copy the launcher files into the binary folder as described in `INSTALL-WINDOWS.md`. See [the import review](docs/REVIEW-v0.8.0.md) for checks repeated after the VM export.
