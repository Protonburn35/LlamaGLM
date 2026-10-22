# LlamaGLM

GLM-5.3-Flash (`unsloth/GLM-5.3-Flash-GGUF`, `UD-IQ3_XXS`, 112.1 GiB) across three Windows 11 PCs on a LAN, using upstream llama.cpp (b11780) with `ggml-cuda` and `ggml-rpc` plus a small set of safety patches and launchers.

**Status: no successful inference on the real PCs yet.** Version 0.8.0 fixes the defects found in the 0.7.0 release (see `HANDOFF.md`). It was built for Windows on 2026-10-22 and has not run on the hardware yet.

**Speed:** estimated 1.5-4 tokens/s output and 15-50 tokens/s prompt processing. These speed estimates have not been measured on the three PCs; see [`LIMITATIONS.md`](LIMITATIONS.md).

| PC | GPU / RAM | Role | Holds (default plan) |
|---|---|---|---|
| A 192.168.50.206 | RTX 4090 24 GB / 32 GB | `llama-server` API on `127.0.0.1:8082`, model files | layers 0-15 in RAM (mmap), 16-22 on the 4090 |
| B 192.168.50.129:50052 | RTX 4060 Laptop 8 GB / 32 GB | RPC worker (GPU + RAM) | 23-30 in RAM, 31-32 on the GPU |
| C 192.168.50.143:50053 | RTX 5070 Ti 16 GB / 32 GB | RPC worker (GPU + RAM) | 33-40 in RAM, 41-44 + output on the GPU |

The model does not fit in the 48 GB of the three GPUs, so each worker also offers its RAM, and Stage A pages part of its layers from its SSD. All weights keep their original quantization; all routed experts are computed; activations cross PCs in F32.

- First run: [`FIRST-RUN.md`](FIRST-RUN.md)
- Install details: [`INSTALL-WINDOWS.md`](INSTALL-WINDOWS.md)
- Status, review of 0.7.0, next steps: [`HANDOFF.md`](HANDOFF.md)
- Memory plan with exact sizes: [`docs/MEMORY_PLAN.md`](docs/MEMORY_PLAN.md)
- Limitations, speed estimates, hardware vs. implementation: [`LIMITATIONS.md`](LIMITATIONS.md)
- Open issues: [`docs/OPEN_ISSUES.md`](docs/OPEN_ISSUES.md)

Not supported yet: MTP/NextN speculation, vision, context beyond the 8K test setting.

The GitHub checkout contains the engine at its root and the launchers in `scripts/`. After building, copy the launcher files into the binary folder as described in `INSTALL-WINDOWS.md`. See [the import review](docs/REVIEW-v0.8.0.md) for checks repeated after the VM export.
