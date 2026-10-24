# LlamaGLM v0.8.1 - limitations

Status: built, **not yet run on the three PCs**. All speed figures below are **estimates from bandwidth arithmetic** (real tensor sizes from the GGUF headers, published bandwidth of the parts). They are not measurements. The first real run (`smoke-test.ps1`) replaces them.

Each limitation is marked:
- **HW** = hardware limit: no software on these three PCs can remove it.
- **IMPL** = implementation: a different engine or configuration could remove or reduce it.
- **BOTH** = partly each.

## 1. Speed targets: 500 tok/s prefill, 40 tok/s output

| | Target | This build (estimate) | Best any software could do on these PCs (estimate) | Verdict |
|---|---|---|---|---|
| Output (decode, one request) | 40 tok/s | about **1.5-4 tok/s** (ceiling 8 tok/s with perfect bandwidth) | about **23-25 tok/s** realistic, **26-32 with MTP** (38-40 theoretical), with a Strata-style expert cache that follows the conversation; about 17 tok/s with a fixed hot-expert list. Based on GLM's measured expert usage | **HW: not reachable on these PCs.** With a 16 GB desktop GPU in place of the laptop's: 29-33 tok/s, 32-42 with MTP |
| Prefill (prompt processing) | 500 tok/s | about **15-50 tok/s** | roughly **500-1000 tok/s**, only for long prompts (several thousand tokens) and with a very different engine | **IMPL: reachable in principle for long prompts, not with this engine** |

### Output: the hardware ceiling, and what expert caching changes

Every generated token reads all weights that the token uses:
- non-routed weights (attention, shared expert, dense layers, output head): **6.68 GiB, read every token**;
- routed experts: 8 of 288 per layer = **2.84 GiB per token** (of 102.3 GiB in total).

Speed is set by where those bytes live:

| Memory | Read speed | Size |
|---|---|---|
| RTX 4090 / 5070 Ti / 4060 Laptop VRAM | about 1000 / 900 / 256 GB/s | 24 / 16 / 8 GB = 48 GB in total |
| System RAM of each PC (dual-channel DDR5) | about 60-65 GB/s | 32 GB each, about 24 GiB usable |
| Stage A NVMe (pages of mmapped weights) | about 1-5 GB/s | - |

The model needs 109.5 GiB. The GPUs hold at most about 42 GiB of weights and each PC about 24 GiB in RAM, so about two thirds of the routed experts (about 67 GiB) must sit in system RAM, and each expert can live in only one place. (Strata, on one PC with 64 GB, keeps every expert in RAM plus a VRAM copy of the busiest ones; three 32 GB PCs cannot.) Speed then depends on the **hit rate**: the share of a token's experts that are already on a GPU. The best placement keeps every non-routed weight on the GPU of the PC that owns the layer, fills the rest of each GPU with that PC's most-used experts, and lets the CPU compute the missed experts while the GPU computes the cached ones:

| Expert usage | Hit rate A / B / C | Perfect bandwidth | Realistic | Realistic with MTP |
|---|---|---|---|---|
| Even: every expert equally likely (worst case for caching) | 44% / 21% / 34% | 22 tok/s | 13 tok/s | 13-15 tok/s |
| **GLM measured, fixed list of busiest experts** (from Unsloth's imatrix counts) | 63% / 37% / 52% | 27 tok/s | 17 tok/s | 17-19 tok/s |
| **GLM, cache that follows the conversation (Strata-style)**; adaptation gain taken from Qwen | 82-88% / 63% / 73-76% | 38-40 tok/s | 23-25 tok/s | 26-32 tok/s |
| For comparison: Qwen3.8-Flash-Next hit rates as Strata measured them in real use | 91% / 68% / 81% | 44 tok/s | 27 tok/s | 32-36 tok/s |

Assumptions (`tools/decode-speed-model.py`): A runs 20 layers on the 4090 (the 3 dense layers and 17 MoE layers), B 11 on the 4060 Laptop, C 14 plus the output head on the 5070 Ti, the split that fits 24 GiB of RAM per PC; each PC's VRAM holds 44% / 21% / 34% of its experts; 2 LAN hops (none routed through A). Realistic = GPU kernels at 70% and CPU kernels at 60% of memory bandwidth, plus about 2.5 ms per token of launch and synchronisation overhead. MTP = one draft token, 60-80% accepted (GLM has one NextN layer).

40 tok/s (25 ms per token) is not a realistic target on these PCs, even with an expert cache:
- **The 4060 Laptop stage is the slowest part:** 256 GB/s VRAM with room for only a fifth of its experts, about 12 ms of the 40 ms per token. A 16 GB desktop GPU with about 450 GB/s in its place: 29-33 tok/s realistic, 32-42 with MTP.
- **GLM-5.3-Flash moves about 4x more bytes per token than the Qwen model that Strata runs at about 60 tok/s on one RTX 5070** (IQ3_XXS, with MTP). GLM's experts are about 8.7 MiB each (Qwen's about 1.4 MB), 8 per layer in 42 MoE layers, about 3 GB of experts per token, plus 6.7 GiB of non-routed weights read for every token.
- **GLM's expert usage is uneven, but less than Qwen's.** Unsloth publishes the imatrix file it used to make these quants (`imatrix_unsloth.gguf_file`); it counts every expert pick over about 0.8M tokens of Unsloth's calibration text. Caching the busiest 44% of a layer's experts catches 63% of GLM's picks and 72% of Qwen's (21%: 37% vs 47%; 34%: 53% vs 62%). GLM also routes differently: a sigmoid plus a learned per-expert balancing bias (DeepSeek-V3 style, `noaux_tc`), 8 of 288 larger experts, against Qwen's softmax over 512 small experts.
- **The Strata-style row is an estimate on top of that measurement.** A cache that follows the conversation beats a fixed list: on Qwen, Strata measured about 61% hits instead of 34% with 13% of experts cached, and 92% instead of 73% with 45%. The row assumes GLM gains as much from adapting as Qwen does, which is not measured. A route log from the real PCs (which 8 experts each token picks in each layer) would measure it.

The layer stages also cannot run at the same time for one request: token N+1 needs token N, and layer L+1 needs layer L. Splitting each layer across the PCs (tensor parallelism) instead would need about 90 network round trips per token. On 1 GbE under Windows, each round trip costs about 0.2-0.5 ms, so that alone takes 18-45 ms.

Speculative decoding with the built-in MTP layer helps only when most experts are hits: checking 2 tokens at once reads the non-routed weights once but almost twice the experts. With even expert usage the gain is close to none; with GLM's measured usage about 1.05-1.25x (included in the table). The MTP layer (layer 45) needs another 2.6 GiB, which these PCs barely have.

**What would reach 40 tok/s with this model:** one machine with 8-12 channel DDR5 (Threadripper / EPYC class, 128 GB or more, about 250-450 GB/s) plus the 4090. Estimated ceiling there: about 60 tok/s, realistic 30-40. On these PCs, a Strata-class engine for GLM plus a 16 GB desktop GPU in place of the laptop's and MTP gives an estimated 32-42 tok/s, so 40 is at the top of that range. Alternatively, a smaller model or a lower quant (but the project rule is to keep UD-IQ3_XXS).

### Why this build decodes at about 1.5-4 tok/s (implementation, on top of the hardware limit)

The current plan puts **whole layers** in system RAM, including their attention weights. Per token that is 3.3 GiB from Stage A RAM, 1.6 GiB from B RAM and 1.6 GiB from C RAM, about 6.5 GiB in total, instead of the 1.9 GiB of the ideal placement:

| Stage | Read per token | Time at perfect bandwidth |
|---|---|---|
| A RAM (layers 0-15) | 3.29 GiB | 56 ms |
| A RTX 4090 (16-22) | 1.38 GiB | 1.5 ms |
| B RAM (23-30) | 1.58 GiB | 28 ms |
| B RTX 4060 (31-32) | 0.40 GiB | 1.7 ms |
| C RAM (33-40) | 1.58 GiB | 26 ms |
| C RTX 5070 Ti (41-44 + output) | 1.28 GiB | 1.5 ms |
| **Sum** | | **about 120 ms = 8.3 tok/s ceiling** |

On top of that, about 15 GiB of Stage A's 35 GiB do not fit in its RAM and come from its SSD when an expert is not cached. Each token reads about 0.3 GiB of experts that are not in RAM, which adds about 0.1-0.3 s. That is where 1.5-4 tok/s comes from.

**IMPL improvement (not done, not tested):** keep the non-routed weights of every layer on the GPU of the PC that owns the layer and put only routed experts in its RAM (llama.cpp `--override-tensor` / `-ot` patterns with the `RPC0[...]`/`RPC1[...]` buffer types). This reduces system RAM reads per token about 3.5x and removes most of Stage A's SSD paging. With stock RPC, every switch between a worker's GPU and RAM costs a LAN round trip commanded by Stage A, so expect 2-3x rather than 3.5x. The 17-32 tok/s range above needs a custom engine that runs each PC's layers locally (StrataRPC-style stages) with Strata's hot-expert cache.

### Why prefill is about 15-50 tok/s here (mostly implementation)

A prompt processes many tokens per pass (512 per ubatch here), so the weights are read once per pass and the work becomes compute-bound. About 16.7 B parameters are active per token, about 33 GFLOP per token. 500 tok/s therefore needs about 17 TFLOP/s sustained.

- **This build:** 32 of 45 layers are computed by the CPUs (`--no-op-offload`). An 8-core desktop CPU does well under 1 TFLOP/s on these quant types, the three CPUs work one after another, and Stage A also re-reads about 13 GiB from its SSD every ubatch. Estimate: 15-50 tok/s. **IMPL.**
- `-OpOffload` lets the 4090 compute Stage A's RAM layers by copying their weights over PCIe each ubatch. A larger `-ub` (2048) spreads that copy over more tokens. Expected: a moderate gain, perhaps 1.5-2x, not 10x. **IMPL (switch exists, untested).**
- **Stock llama.cpp RPC cannot do the same on the workers.** Weights in a worker's RAM device (RPC1/RPC3) are always computed by that worker's CPU; op offload only moves work from Stage A's own CPU to its own GPU. **IMPL (would need a worker-side change).**
- **Hardware ceiling:** if every PC streamed its RAM-resident experts through its own GPU (4090 PCIe 4.0 x16, 5070 Ti PCIe 5.0 x16, 4060 Laptop PCIe 4.0 x8) with ubatches of about 4096 tokens, and the three PCs worked on different ubatches at the same time, the arithmetic gives several hundred to about 1000 tok/s for long prompts. **HW limits that remain:**
  - **1 GbE.** Each token crosses the LAN as 64 KiB of FP32 activations (4 hyper-connection streams x 4096), so one link carries at most about 1800 tok/s (117 MB/s). If the transfers do not overlap with computation, as in llama.cpp, prefill is capped at **about 890 tok/s with 2 LAN hops** (A->B->C) or **about 600 tok/s with 3 hops**; llama.cpp RPC routes B->C through A, which is 3 hops. 2.5 GbE would raise these caps 2.5x. The FP32 wire format is a project rule; FP16 would halve the traffic.
  - **Short prompts never reach 500 tok/s.** Streaming 20-35 GiB of weights to a GPU takes seconds per ubatch no matter how many tokens it holds, and computing in place on the CPU is slow. A few-hundred-token prompt takes a few seconds before the first token either way.

## 2. Will it work at all

| # | Limitation | Type | Notes |
|---|---|---|---|
| 1 | **No run on the real PCs yet.** v0.8.1 is built with MSVC and nvcc (the upstream-supported toolchain) on GitHub Actions; the earlier v0.8.0 was cross-compiled with clang on Linux. | IMPL | The per-GPU self-test (`test-backend-ops`, about 1-5 min) runs the model's ops on every GPU before any model data moves. If kernels misbehave, the official llama.cpp b11780 build (CUDA 13.4) of the same upstream source runs the same plan: `-BinDir` (FIRST-RUN.md section 6). |
| 2 | Root cause of the earlier "illegal memory access" is unknown. | IMPL / unknown | Fixed suspects: clang finite-math kernels, infeasible memory plan, sysmem fallback, unified memory. A real kernel bug for GLM5 shapes is not excluded. |
| 3 | Answer check is a plausibility test (Paris, 2+2=4). | IMPL | Catches broken kernels (garbage tokens). It does not prove full numerical accuracy; a perplexity comparison against the official build would. |
| 4 | Memory fits are tight: B and C each need about 24 GiB free RAM, the 4090 about 20 GiB free VRAM. | HW | Gate 2 and the worker reserve refuse before anything is allocated. Nothing protects a running model: starting a game or many browser tabs on B or C can push worker weights into the page file (very slow) or fail a CUDA allocation (worker crash). |
| 5 | A lost worker ends `llama-server`. Laptop sleep, a Wi-Fi drop, closing the worker window or a CUDA error all count. | IMPL (llama.cpp RPC design) | Workers restart themselves up to 3 times; Stage A must then be started again. No request recovery. Keep the laptop on mains power, lid-close = do nothing, wired LAN. |
| 6 | A remote GPU aborts on an op it does not support (RPC `supports_op` is an upstream TODO). | IMPL (upstream) | The self-test covers the model's ops on each GPU, including flash attention with K=V=512 for the DSA layers. |
| 7 | Updating the binaries means updating all three PCs (the RPC protocol must match). | IMPL | Includes the upstairs 5070 Ti PC (or use remote desktop). Changing the layer split only touches Stage A's `cluster_config.json`. |

## 3. Operational limits

| # | Limitation | Type | Notes |
|---|---|---|---|
| 8 | First load sends about 56 GiB to the workers: about 9-10 min on 1 GbE. | HW (LAN) | Later loads send only hashes; the workers read weights from `C:\LlamaGLM-Cache` (about 26 / 32 GiB of disk). Stage A also hashes 56 GiB on every load (about 1 min). |
| 9 | Windows prefetch: llama.cpp asks Windows to prefetch all 112 GiB of the mmapped shards at load. | IMPL (upstream) | Not harmful (clean file pages), but it adds SSD reads during load and evicts other cached files on Stage A. |
| 10 | Stage A pages about 15 GiB of weights from its SSD (balanced plan); `laptop-light` pages about 30 GiB. | BOTH | 32 GB RAM per PC is the hardware cause; the whole-layer placement makes it worse (see section 1). Reads only, no SSD wear. |
| 11 | Context is 8K and one request at a time (`-c 8192 -np 1`). | BOTH | KV memory is small (DSA layers only), so 32-64K would probably fit (Gate 2 shows it). But at 15-50 tok/s prefill, a 30K-token agent prompt takes 10-30 min. Nothing here supports 1M context. |
| 12 | No MTP/NextN speculation, no vision. | IMPL | MTP gains about 1.05-1.25x, and only with a hot-expert cache (section 1). |
| 13 | RPC has no authentication or encryption. | IMPL (upstream) | Firewall rules allow only 192.168.50.206. Never forward 50052/50053. The API listens on 127.0.0.1:8082 only. |
| 14 | CPU backend uses AVX2 only. The 7800X3D and 9800X3D also have AVX-512. | IMPL | Maybe 10-20% more CPU speed with `GGML_CPU_ALL_VARIANTS` (not used yet: the launchers expect a single `ggml-cpu.dll`). |
| 15 | Worker threads = all physical cores. On a laptop CPU with E-cores this can be slower than P-cores only. | IMPL (config) | Set `stage_b.threads` in `cluster_config.json`. |
| 16 | Windows PowerShell 5.1 not tested (scripts tested with PowerShell 7 on Linux). | IMPL | If a 5.1 error appears, run the same script with `pwsh`. |
| 17 | Worker weight caches are never cleaned. | IMPL | Delete `C:\LlamaGLM-Cache` to reclaim space after a model change. |
| 18 | v0.8.0 only: clang could not unroll some loops that nvcc unrolls (7148 `loop not unrolled` warnings, almost all from the flash-attention and quantized matrix kernel templates). | IMPL (toolchain) | Gone in v0.8.1, which uses nvcc. Correctness was not affected. |

## 4. Summary

- **Correctness:** the build is designed to load the full model across the three PCs without freezing Windows, and to stop early with a clear message if something does not fit or a GPU kernel is broken. Whether it generates correct text is unproven until `smoke-test.ps1` passes on the PCs.
- **40 tok/s output is beyond what this hardware can do** with this model and quant (3 x 32 GB RAM + 48 GB VRAM over 1 GbE). GLM's expert usage, measured from Unsloth's imatrix counts, is uneven enough for an expert cache to help, but less than Qwen's. A Strata-style engine (hot experts in VRAM, missed experts on the CPU, MTP) is estimated at 23-25 tok/s, 26-32 with MTP; about 17 tok/s with a fixed hot-expert list. With a 16 GB desktop GPU in place of the laptop's: 32-42 tok/s with MTP. This build's layout is far below that (about 1.5-4 tok/s); a per-tensor layout could gain 2-3x.
- **500 tok/s prefill is an implementation limit today** (estimate 15-50 tok/s). The hardware could reach it only for long prompts, with GPU weight streaming on all three PCs and stages that overlap; without overlap, 1 GbE alone caps prefill at about 600-890 tok/s (2.5 GbE recommended). Short prompts stay at a few seconds before the first token.
