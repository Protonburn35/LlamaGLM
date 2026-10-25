# Speed research (2026-10-25)

Research only: no engine or launcher code was changed. Inputs: the user's v0.8.2 logs from 2026-10-25 (HAR = A, PROART = B, ASUSPC = C; balanced plan, `-Prefetch -Timing` on A, `-Timing` on both workers, `perf-log.ps1 -Role A`), the engine source, and local runs of the tiny GLM5-next fixture over two local RPC workers. Constraints kept: same model and mixed quantization, all 8 selected experts, normal nonzero-temperature sampling. Numbers marked "estimate" are arithmetic, not measurements. The earlier analysis is [SPEED_RESEARCH-2026-10-24.md](SPEED_RESEARCH-2026-10-24.md).

## 1. What the logs measure

| Item | Value | Source |
|---|---|---|
| Prompt | 215 tokens in 58.9 s (3.65 tok/s); the 203-token batch took 51.9 s | server log |
| Generation | 1672 tokens, 565 ms/token (1.77 tok/s); 1.71 tok/s on 2026-10-24 without prefetch, over a shorter answer | server log |
| Graph | 14758 nodes, 6 splits, 1657 of 1672 graphs reused | server log |
| Fused ops | `fused DeepSeek V4 HC post not supported, set to disabled`, triggered by layer 17 (last CPU layer) | server log |
| Decode, per token (steady state) | A CPU (layers 0-17) 190-290 ms; 4090 (18-24) 64 ms; B RAM 62 ms; B GPU 51.5 ms; C RAM 61-63 ms; C GPU, logits and sampling ("rest") 37-61 ms | `sched_profile` lines |
| Worker compute per graph | B RAM 39.4 ms, B GPU 42.4 ms, C RAM 41.1 ms, C GPU 48.1 ms (averages that include the prompt) | `[stats] busy` |
| Nodes per layer | about 330 on every device | `[stats]` nodes/graphs |
| Bytes per hop | 384 KiB per token from each of B RAM, B GPU and C RAM, constant over the whole answer; 605 KiB of logits from C GPU | `[stats] sent` deltas |
| A during generation | SSD 250-390 MB/s (about 72k pages/s), read queue 0.5-0.8, 150-300 MB RAM available, llama-server 680-700% CPU (48% of the 7800X3D) | `perf-a-*.csv` |
| A during the prompt | SSD 590 MB/s, read queue 2.3 | same |
| Free memory at the end | 4090 4.1 GiB, B RAM 5.0, B GPU 0.7, C RAM 5.9, C GPU 3.1 | memory breakdown |

## 2. Findings

### 2.1 The hyper-connection post step runs unfused everywhere

GLM5-next mixes four residual streams around every attention and FFN block. The fused `DSV4_HC_POST` op does that mixing in one node. `llama_context::resolve_fused_ops` disables it for the whole model if any layer's fused node lands on a device other than that layer's own device, assuming missing support. Here the cause is the scheduler, not support: layer 17's post step has no weights, and the scheduler's GPU expansion pass assigns weightless nodes at the CPU-to-4090 boundary to the 4090. The CPU, CUDA and RPC backends all implement `DSV4_HC_POST`.

The unfused fallback costs time in two ways.

- **Graph size.** On the tiny fixture the graph has 319 nodes with the fused op and 631 without it; the real model's 14758 nodes (about 330 per layer) match the unfused shape. Fusing should roughly halve the node count (estimate).
- **Round trips between PCs.** The unfused mixing reads 16 views of the previous layer's output. The scheduler copies each view as a separate input, so every device boundary moves 18 tensors (the 64 KiB output, its 64 KiB norm, and 16 views of 16 KiB) instead of 2. This is exactly the 384 KiB per hop in the worker stats, and it was reproduced on the fixture (split inputs: 20 unfused, 4 fused, with 2 non-hidden inputs in both). Each RPC input copy is synchronous. Between two devices of the same worker, the client first tries `COPY_TENSOR` (it fails for RAM to GPU), then `GET_TENSOR` through A, then sets the tensor. That is about 36 round trips at the B RAM to B GPU boundary, 18 from B to C, and 36 at C RAM to C GPU: about 90 per token.

Fetch time minus worker compute time gives 24 ms, 9 ms and 21 ms at these three boundaries: about 0.5-0.65 ms per round trip and about 54 ms per token in total. Bandwidth is not the limit: about 3.2 MiB per token is 45-50 Mbit/s at this speed, which matches the user seeing little network activity.

### 2.2 GPU layers are slower per layer than CPU layers

Per layer, the 4090 takes 9.2 ms, the 5070 Ti about 10 ms and the 4060 Laptop 21 ms, against 5.6-5.9 ms for the worker CPU layers. Reading one token's weights for a layer (about 0.1 GiB) takes well under 1 ms on any of these GPUs. Per graph node this is 28 µs on the 4090, 35 µs on the 5070 Ti and 65 µs on the laptop, so a fixed cost per kernel dominates. The 203-token prompt batch took 480 ms on the 4090, only 7.5 times the 1-token step.

Two causes fit, and the logs cannot separate them.

- **Kernel launch overhead.** If CUDA graphs are not taking effect, every kernel goes through the Windows WDDM launch path. The code suggests graphs should work: graph reuse keeps each split's `uid`, and workers replay stored graphs through `GRAPH_RECOMPUTE`. The logs have no CUDA graph debug lines to confirm it.
- **Low clocks.** The GPUs are busy about 10% of each token, which may keep them in a low-clock state.

Halving the node count (2.1) lowers the per-node part either way. A clock log (`nvidia-smi`) and one run with "Prefer maximum performance" would separate the two causes.

### 2.3 A still pages experts from its SSD

A maps 41.2 GiB of weights with about 25 GiB of RAM available for them. During generation it reads about 165 MB per token from the SSD (estimate: 300 MB/s at 1.77 tok/s) at a queue depth below 1. That is roughly 150 ms of the 250 ms A spends on its 18 CPU layers; the workers' CPU layers take about 5.7 ms each. Prefetch made the reads larger (about 400 KB per read instead of 32 KB faults) but not concurrent. Generation went from 586 to 565 ms/token, about 4%, and the longer answer makes that comparison approximate.

Microsoft documents `PrefetchVirtualMemory` as a hint that can fail completely or partly under low-memory conditions, and A runs with 150-300 MB available. For the prompt, A must read about 16 GiB of non-resident experts (estimate: 41 GiB mapped, about 25 GiB available) at 590 MB/s, roughly 30 of the 52 s; A's CPU split took 31.6 s of that batch.

The 7800X3D's 50% load is mostly worker threads spinning at barriers while one thread waits for the SSD, not computation. llama-server shows about 7 busy threads for the whole token, although A computes for less than half of it.

## 3. Options, ranked

| # | Option | Kind | Expected effect (estimate) | Risk |
|---|---|---|---|---|
| 1 | Keep the fused HC post on when the layer's own device supports it (the mismatch only means the scheduler placed the node elsewhere). | engine fix in `resolve_fused_ops`, opt-in switch at first | about 80 fewer round trips (about 45 ms/token); about half the graph nodes on every device, which may cut GPU time substantially if the per-node cost dominates; CPU layers also run fewer ops | low: same op the model uses whenever the check passes; fused and unfused results differ only by float rounding |
| 2 | `-LocalCopy` on the workers (already in v0.8.2) | test, no build | today about 36 fewer round trips (15-20 ms/token); after option 1 about 2 ms | low; not yet run on GPUs |
| 3 | GPU clocks: NVIDIA Control Panel, Program Settings, "Power management mode: Prefer maximum performance" for `llama-server.exe` (A) and `ggml-rpc-server.exe` (B, C); laptop on AC in its performance mode; log clocks with `nvidia-smi` | test, no build; logging can join `perf-log.ps1` | unknown; up to most of the 150 ms GPU time if low clocks are the cause | none |
| 4 | Move layers off A into measured free memory: `-Plan tight` (one more layer on the 4090, in v0.8.2) plus a plan with one more layer in C's RAM. B's RAM has room for a layer only at its 2.5 GiB margin; C's GPU has none. | config, Gate 2 decides | 18 to 16 CPU layers on A, 5.2 GiB less mapped (estimate 30-60 ms/token, and a shorter prompt) | low; Gate 2 and the worker reserves refuse plans that do not fit |
| 5 | Copy only `view_src` when several views of one tensor cross a split | engine (`ggml-backend.cpp` scheduler) | covers other models and prompt graphs; nothing extra for decode once option 1 is on | medium: core scheduler code |
| 6 | App-managed expert reads on A (overlapped, unbuffered, many in flight, into a fixed RAM cache) instead of mmap faults; an upstream Windows user measured 2.8 GB/s this way against 377 MB/s through mmap | engine, large | up to about 100 ms/token and 20 s per prompt (estimate) | high effort; a separate project |
| 7 | Backend sampling (`-bs`, upstream, experimental) on C's GPU | test/launcher option | saves the 605 KiB logits transfer, about 5 ms/token on 1 GbE | experimental; check the client's samplers are supported |
| 8 | MTP (`-Mtp`), re-measured after 1-4 | test | 81% acceptance with one draft token on 2026-10-24; gains more once fixed per-step costs shrink | needs the MTP plan's memory |

Options 1-4 together could plausibly reach 300-400 ms/token (2.5-3.3 tok/s) before MTP (estimate). The range is wide mainly because the GPU cause in 2.2 is unknown.

## 4. Proposed v0.8.3 (every item opt-in)

1. `-FusedHc` on A: `resolve_fused_ops` treats a fused node on another device as a mismatch only if the layer's own device cannot run the op (`ggml_backend_dev_supports_op`). The server log shows whether the fused HC post is enabled. Check locally that the fixture's text is identical, or differs only by expected float rounding, with fused and unfused HC post, and that the boundary carries 2 hidden-state inputs.
2. `perf-log.ps1` also logs `nvidia-smi` pstate, SM and memory clocks, utilization and power each second, on A, B and C.
3. A plan that adds one layer to the 4090 and one to C's RAM.

Test session (one sitting): workers with `-Timing -LocalCopy`, NVIDIA "Prefer maximum performance" on all three PCs, `perf-log.ps1` on all three, then on A: (a) `-Timing -Prefetch -Plan tight`, (b) the same plus `-FusedHc`, (c) (b) with the new plan. Compare `sched_profile`, worker `busy` and the GPU clocks.

## Sources

- Engine source: `src/llama-context.cpp` (`resolve_fused_ops`), `src/models/glm5-next.cpp` (HC pre/post, DSA selection), `ggml/src/ggml-backend.cpp` (split inputs, `ggml_backend_tensor_copy`), `ggml/src/ggml-rpc/ggml-rpc.cpp` (`cpy_tensor`, `GRAPH_RECOMPUTE`), `ggml/src/ggml-cuda/ggml-cuda.cu` (CUDA graph warmup and reuse).
- [PrefetchVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory) (Microsoft Learn): a strong hint that can completely or partially fail under low-memory conditions.
- [llama.cpp discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324): overlapped reads against mmap paging for MoE experts on Windows.
