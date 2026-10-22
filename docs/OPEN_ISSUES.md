# Open issues

**No inference has succeeded on the real hardware.** Gates 0-2 and the tiny-model RPC path pass on the Linux VM; gates 3-5 need the user's PCs.

| # | Issue | Status | What to do |
|---|---|---|---|
| 1 | Windows CUDA package v0.8.0 built (2026-10-22) but not run on the PCs | open | Run `FIRST-RUN.md`. Do not use `dist/LlamaGLM-v0.7.0-*.zip`. |
| 2 | Root cause of the illegal memory access | unknown | Candidates: clang finite-math device code (proven defect, fixed in 0.8.0), WDDM sysmem fallback / memory pressure / TDR (disable the fallback policy), unified memory env var (removed), a real kernel bug for GLM5 shapes. The per-GPU CUDA self-test, `-Diagnostic` and the `-BinDir` upstream baseline separate them. |
| 3 | ggml-cuda compiled by clang instead of nvcc | risk | Not an upstream-supported configuration for ggml-cuda. The official nvcc/MSVC build of the same source (b11780, CUDA 13.4, sm_89 + sm_120a) is the reference: `FIRST-RUN.md` section 6. |
| 4 | RPC devices claim support for every op (`supports_op` is an upstream TODO) | upstream limitation | A local GPU falls back to the CPU for an unsupported op; a remote GPU aborts the worker instead. CUDA flash attention for K=V=512 (GLM5 DSA) requires `gqa_opt_applies`; if a DSA layer on B or C hits an unsupported shape, that worker exits with `op not supported`. The self-test runs `FLASH_ATTN_EXT` with `hsk=512` on every GPU first. |
| 5 | Worker loss aborts the coordinator | upstream behaviour | `Remote RPC server crashed or returned malformed response` ends `llama-server`. Workers restart themselves (3 times); then start Stage A again. No request recovery. |
| 6 | Stage A pages ~15 GiB of mmapped weights from its SSD (balanced plan) | by design | The model does not fit in 3 x (GPU + RAM) without it. Prompt processing reads the whole CPU part; decode reads about 8/288 of each MoE layer's experts per token. Speed is unknown. |
| 7 | Laptop runs 10 layers in the balanced plan (2 on GPU, 8 in RAM) | trade-off | `-Plan laptop-light` = 3 layers on the laptop, about 15 GiB more SSD reading on Stage A and ~26 GiB free RAM needed on C. |
| 8 | CUDA compute buffer sizes and free VRAM with the desktop running | unmeasured | Gate 2 prints them before every load and refuses a plan that does not fit. |
| 9 | Patched system header `__clang_cuda_cmath.h` on the build VM | build fragility | See `BUILD_LOG.md`. Reinstalling `libclang-common-20-dev` breaks the cross build. |
| 10 | Windows PowerShell 5.1 not tested | risk | Scripts parse in pwsh 7 and avoid PS7-only syntax. If a 5.1 error appears, `pwsh` (PowerShell 7) can run the same scripts. |
| 11 | GLM chat template always starts with `<think>`; `special_eot_id`/`special_eom_id` warning at load | to observe | Gate 4 accepts text in `reasoning_content`. Check that generation stops at the end-of-turn token in the first real run. |
| 12 | MTP / NextN speculation, vision projector, context > 8K | not started | Layer 45 is skipped. Nothing here claims 1M context. |
| 13 | Worker caches (`C:\LlamaGLM-Cache`) are unbounded | operational | About 26 GiB on B and 32 GiB on C. Delete the folder to reclaim space; the next load re-sends the weights. |
| 14 | RPC transport has no authentication | upstream design | Firewall rules restrict the ports to 192.168.50.206. Never forward 50052/50053 on the router. |
| 15 | CPU backend is AVX2 level only | performance | All three CPUs (Zen 4 / Zen 5 desktops, 2023+ laptop) support AVX2. The 7800X3D and 9800X3D also have AVX-512; `-DGGML_CPU_ALL_VARIANTS=ON` (upstream release style, one `ggml-cpu-*.dll` per CPU level) would use it, but is untested with MinGW. Later. |
| 16 | Speed: estimated 1.5-4 tok/s output, 15-50 tok/s prefill; user target 40 / 500 | open | 40 tok/s output is beyond this hardware (ceiling about 23 tok/s even with ideal software); prefill is limited mostly by this implementation. Details and options: `LIMITATIONS.md`. |
| 17 | clang leaves some `#pragma unroll` loops rolled (7148 warnings, flash-attention and MMQ templates) | performance | Results are not affected; those kernels may be slower than nvcc's. Compare with `-BinDir` timing once inference works. |
