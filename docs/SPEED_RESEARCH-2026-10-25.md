# Speed research (2026-10-25)

Research only: no engine or launcher code was changed. Inputs: the user's v0.8.2 logs from 2026-10-25 (HAR = A, PROART = B, ASUSPC = C; balanced plan, `-Prefetch -Timing` on A, `-Timing` on both workers, `perf-log.ps1 -Role A`), the engine source, and local runs of the tiny GLM5-next fixture over two local RPC workers. Constraints kept: same model and mixed quantization, all 8 selected experts, normal nonzero-temperature sampling. Numbers marked "estimate" are arithmetic, not measurements. The earlier analysis is [SPEED_RESEARCH-2026-10-24.md](SPEED_RESEARCH-2026-10-24.md).

## 1. What the logs measure

| Item | Value | Source |
|---|---|---|
| Prompt | 215 tokens in 58.9 s (3.65 tok/s); the 203-token batch took 51.9 s | server log |
| Generation | 1672 tokens, 565 ms/token (1.77 tok/s); 1.71 tok/s on 2026-10-24 without prefetch, over a shorter answer | server log |
| Stalls | 51 of 267 three-second samples below 1.3 tok/s, four near 0.3 tok/s; median 1.98 tok/s against a mean of 1.79 (2026-10-24: 13 of 72 samples) | `tg_3s` values |
| A memory | 40.2 GiB (41177 MiB) of weights memory-mapped; 23.8 GiB of RAM free before loading | server log |
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

GLM5-next mixes four residual streams around every attention and FFN block. The fused `DSV4_HC_POST` op does that mixing in one node. `llama_context::resolve_fused_ops` disables it for the whole model if any layer's fused node lands on a device other than that layer's own device, assuming missing support. Here the cause is the scheduler, not support: layer 17's post step has no weights, and the scheduler's GPU expansion pass assigns weightless nodes at the CPU-to-4090 boundary to the 4090. The CPU and CUDA backends both implement `DSV4_HC_POST`; an RPC device reports every op as supported and runs it on the worker's CPU or CUDA backend.

The false positive reproduces locally. With the tiny fixture's layer 0 on the CPU and layer 1 on an RPC worker backed by a CPU, the same warning appears and the fused HC post is disabled; with every layer on the worker it stays enabled. On the fixture, fused and unfused HC post give identical text (48 tokens, temperature 1, fixed seed, over two RPC workers), and the unfused worker graphs have twice the nodes (15043 against 7399 on one worker).

The unfused fallback costs time in two ways.

- **Graph size.** On the tiny fixture the graph has 319 nodes with the fused op and 631 without it; the real model's 14758 nodes (about 330 per layer) match the unfused shape. Fusing should roughly halve the node count (estimate).
- **Round trips between PCs.** The unfused mixing reads 16 views of the previous layer's output. The scheduler copies each view as a separate input, so every device boundary moves 18 tensors (the 64 KiB output, its 64 KiB norm, and 16 views of 16 KiB) instead of 2. This is exactly the 384 KiB per hop in the worker stats, and it was reproduced on the fixture (split inputs: 20 unfused, 4 fused, with 2 non-hidden inputs in both). Each RPC input copy is synchronous. Between two devices of the same worker, the client first tries `COPY_TENSOR` (it fails for RAM to GPU), then `GET_TENSOR` through A, then sets the tensor. That is about 36 round trips at the B RAM to B GPU boundary, 18 from B to C, and 36 at C RAM to C GPU: about 90 per token.

Fetch time minus worker compute time gives 24 ms, 9 ms and 21 ms at these three boundaries: about 0.5-0.65 ms per round trip and about 54 ms per token in total. Bandwidth is not the limit: about 3.2 MiB per token is 45-50 Mbit/s at this speed, which matches the user seeing little network activity.

### 2.2 GPU layers are slower per layer than CPU layers

Per layer, the 4090 takes 9.2 ms, the 5070 Ti about 10 ms and the 4060 Laptop 21 ms, against 5.6-5.9 ms for the worker CPU layers. Reading one token's weights for a layer (about 0.1 GiB) takes well under 1 ms on any of these GPUs. Per graph node this is 28 µs on the 4090, 35 µs on the 5070 Ti and 65 µs on the laptop, so a fixed cost per kernel dominates. The 203-token prompt batch took 480 ms on the 4090, only 7.5 times the 1-token step.

Several causes fit, and the logs cannot separate them.

- **Kernel launch overhead.** If CUDA graphs are not taking effect, every kernel goes through the Windows WDDM launch path. The code suggests graphs should work: graph reuse keeps each split's `uid`, and workers replay stored graphs through `GRAPH_RECOMPUTE`. The expert matrix multiply does not switch graphs off here: for one-token steps it uses MMVQ without a stream sync on Ada and Blackwell (`get_mmvq_mmid_max_batch`). The delta-net layers use the fused kernel (`fused_gdn_ar` is on by default without a probe). The logs have no CUDA graph debug lines to confirm graphs are active.
- **Low clocks.** The GPUs are busy about 10% of each token, which may keep them in a low-clock state.
- **A few slow kernels**, which only per-op GPU timing would show.
- **Laptop specifics.** The 4060 Laptop also drives a display (`Disp.A On` in `nvidia-smi`), and its event log has `nvlddmkm` error events (ID 153, usually GPU resets) on 2026-10-24, none during this run.

Halving the node count (2.1) lowers the per-node part in every case. A clock log (`nvidia-smi`) and one run with "Prefer maximum performance" would test the clock theory. One run with `GGML_CUDA_DISABLE_GRAPHS=1` set before starting A would show whether CUDA graphs were active: if the 4090's time does not change, they were not. "Hardware-accelerated GPU scheduling" in Windows display settings also affects launch cost.

### 2.3 A still pages experts from its SSD

A maps 40.2 GiB of weights and had 23.8 GiB of RAM free before loading. During generation it reads about 165 MB per token from the SSD (estimate: 300 MB/s at 1.77 tok/s) at a queue depth below 1. That is roughly 150 ms of the 250 ms A spends on its 18 CPU layers; the workers' CPU layers take about 5.7 ms each. With prefetch, reads average about 430 KB (72k pages over about 685 reads per second). A page fault reads clusters of about 32 KiB; this is documented, not measured, since there were no counters on 2026-10-24. The reads are still not concurrent. Generation went from 586 to 565 ms/token, about 4%, and the longer answer makes that comparison approximate.

Microsoft documents `PrefetchVirtualMemory` as a hint that can fail completely or partly under low-memory conditions, and A runs with 150-300 MB available. For the prompt, A must read about 16 GiB of non-resident experts (estimate: 40.2 GiB mapped, 23.8 GiB free) at 590 MB/s, roughly 30 of the 52 s; A's CPU split took 31.6 s of that batch.

The 7800X3D's 50% load is most likely spinning, not computation. llama-server shows about 7 busy threads for the whole token, although A computes for less than half of it. ggml's CPU threads spin at barriers while one thread waits on a page fault. After each graph, each of the 7 worker threads also polls for new work for up to 1024·128·50 pause instructions before sleeping (default `--poll 50`).

### 2.4 Generation stalls on A

About 19% of the three-second `tg_3s` samples are below 1.3 tok/s, and four are near 0.3 tok/s. Without them generation would run at about 1.98 tok/s instead of 1.79 (about 10%). 2026-10-24 had the same share.

The stalls are on A. Per 30-s window, the workers' fetch times vary by a few ms and "rest" by about 15 ms. A's CPU split ranges from 193 to 292 ms and accounts for most of the change in ms/token (over the 30 windows after the first, ms/token minus A's CPU split stays at 282-318 ms, mostly 295-311). During stall windows A reads less from its SSD, not more (408 against 685 reads/s, 221 against 326 MB/s), and its CPU use is slightly higher. That points to Windows memory management under A's 150-300 MB of free RAM (for example working-set trimming or memory compression) rather than the SSD itself. This is not proven. Freeing memory on A (option 4) should help both the paging and the stalls; per-window maximum times in the opt-in timing would locate them.

## 3. Options, ranked

| # | Option | Kind | Expected effect (estimate) | Risk |
|---|---|---|---|---|
| 1 | Keep the fused HC post on when the layer's own device supports it (the mismatch only means the scheduler placed the node elsewhere). | engine fix in `resolve_fused_ops`, opt-in switch at first | about 80 fewer round trips (about 45 ms/token); about half the graph nodes on every device, which may cut GPU time substantially if the per-node cost dominates; CPU layers also run fewer ops | low: same op the model uses whenever the check passes; fused and unfused results differ only by float rounding |
| 2 | `-LocalCopy` on the workers (already in v0.8.2) | test, no build | today about 36 fewer round trips (15-20 ms/token); after option 1 about 2 ms | low; not yet run on GPUs |
| 3 | GPU clocks: NVIDIA Control Panel, Program Settings, "Power management mode: Prefer maximum performance" for `llama-server.exe` (A) and `ggml-rpc-server.exe` (B, C); laptop on AC in its performance mode; log clocks with `nvidia-smi` | test, no build; logging can join `perf-log.ps1` | unknown; up to most of the 150 ms GPU time if low clocks are the cause | none |
| 4 | Move layers off A into measured free memory: `-Plan tight` (one more layer on the 4090, in v0.8.2) plus a plan with one more layer in C's RAM. B's RAM has room for a layer only at its 2.5 GiB margin; C's GPU has none. | config, Gate 2 decides | 18 to 16 CPU layers on A, 5.1 GiB less mapped (estimate 30-60 ms/token, a shorter prompt, possibly fewer stalls) | low; Gate 2 and the worker reserves refuse plans that do not fit |
| 5 | Copy only `view_src` when several views of one tensor cross a split | engine (`ggml-backend.cpp` scheduler) | covers other models and prompt graphs; nothing extra for decode once option 1 is on | medium: core scheduler code |
| 6 | App-managed expert reads on A (overlapped, unbuffered, many in flight, into a fixed RAM cache) instead of mmap faults; an upstream Windows user measured 2.8 GB/s this way against 377 MB/s through mmap | engine, large | up to about 100 ms/token and 20 s per prompt (estimate) | high effort; a separate project |
| 7 | Backend sampling (`-bs`, upstream, experimental) on C's GPU | test/launcher option | saves the 605 KiB logits transfer, about 5 ms/token on 1 GbE | experimental; check the client's samplers are supported |
| 8 | MTP (`-Mtp`), re-measured after 1-4 | test | 81% acceptance with one draft token on 2026-10-24; gains more once fixed per-step costs shrink | needs the MTP plan's memory |
| 9 | Stall diagnostics: the slowest graph per window in `sched_profile` and the worker `[stats] busy` line | engine, inside the existing opt-in timing | locates the stalls in 2.4 (worth about 10%) | none when off |
| 10 | Prefetch from several threads, or the next layers' experts ahead of time | engine, small to medium | uncertain: Windows may drop the hints at A's free-memory level | low |
| 11 | Non-routed weights of A's CPU layers on the 4090, experts in A's RAM (`--n-cpu-moe`-style placement, optionally with upstream `--moe-cache-mib` for hot experts) | config/launcher | only after 2.2 is fixed: it adds GPU work and CPU-GPU handoffs per layer | medium |

Options 1-4 together could plausibly reach 300-400 ms/token (2.5-3.3 tok/s) before MTP (estimate). The range is wide mainly because the GPU cause in 2.2 is unknown.

## 4. Proposed v0.8.3 (every item opt-in)

1. `-FusedHc` on A: `resolve_fused_ops` treats a fused node on another device as a mismatch only if the layer's own device cannot run the op (`ggml_backend_dev_supports_op`). The server log shows whether the fused HC post is enabled. Check locally that the fixture's text is identical, or differs only by expected float rounding, with fused and unfused HC post, and that the boundary carries 2 hidden-state inputs.
2. `perf-log.ps1` also logs `nvidia-smi` pstate, SM and memory clocks, utilization and power each second, on A, B and C.
3. A plan that adds one layer to the 4090 and one to C's RAM.
4. Stall diagnostics (option 9) inside the existing `-Timing` switches.

Test session (one sitting): workers with `-Timing -LocalCopy`, NVIDIA "Prefer maximum performance" on all three PCs, `perf-log.ps1` on all three, then on A: (a) `-Timing -Prefetch -Plan tight`, (b) the same plus `-FusedHc`, (c) (b) with the new plan. Compare `sched_profile`, worker `busy` and the GPU clocks. Optional, if time allows: (b) again with `GGML_CUDA_DISABLE_GRAPHS=1` set in A's window, to see whether CUDA graphs were active.

## 5. What v0.8.3 implemented

See [TEST-PLAN-v0.8.3.md](TEST-PLAN-v0.8.3.md).

- **Option 1** as `-FusedHc`: `LLAMA_FUSED_HC_FIX=1` makes `resolve_fused_ops` accept a fused node on another device when both devices support the op.
- **Option 9** in `-Timing`: the slowest graph per window, with run + fetch per device, and the slowest worker graph since the last `[stats]` line.
- **GPU clock logging** in `perf-log.ps1`.
- **Plans:** `relief` (option 4) and `gpu-attn` (option 11), with `-MoeCacheMib`.
- **`-NoCudaGraphs`** for the graph diagnostic.

Not implemented:
- **Pinning the HC pre norm to the next layer's device.** Tried as a way to bring a hop down to one hidden-state copy, but the scheduler then copies the reshaped view of the residual instead, so it stays at 2 copies. Removed.
- **Options 5-7 and 10.**

Found while testing: the v0.8.2 timing report at scheduler shutdown read device names from backends that llama.cpp frees first. It crashed in about half of the local `-Timing` runs; now the names are copied at creation.

## 6. Plan after the v0.8.3 test

What each result of [TEST-PLAN-v0.8.3.md](TEST-PLAN-v0.8.3.md) leads to:

| Result | Next step |
|---|---|
| `-FusedHc` lowers the fetch overhead (about 45 ms expected) and GPU times, with normal answers | make it the default in the launchers, keeping a switch to turn it off |
| `gpu-*.csv` shows low clocks or a high pstate number during generation, and "Prefer maximum performance" fixes it | make it a setup step in FIRST-RUN.md |
| GPU layers still take several ms each with the fused op and full clocks | add per-op CUDA timing (opt-in, events around each node with CUDA graphs off) to find the slow kernels |
| `-NoCudaGraphs` does not change the 4090's time | find out why graphs do not apply (the `GGML_CUDA_GRAPH_DEBUG` log with `-v`) |
| `slowest graph` puts the stalls in A's CPU split | treat them as memory pressure on A: `relief`, then option 6 |
| `gpu-attn` is faster | adopt it; tune `n_cpu_moe` and the MoE cache size |
| `relief` passes Gate 2 and is faster | make it the default plan |

### Option 6 sketch: expert reads managed by llama.cpp on A

The largest remaining item if paging stays the main cost.

- **Cache:** a fixed RAM cache of expert slabs, for example 16-18 GiB, organized like upstream `llama-moe-cache`: an SLRU keyed by (layer, expert), with a slot map read by `get_rows`.
- **Reads:** misses are read straight from the GGUF file with overlapped, unbuffered I/O, several MiB per request and 16 or more requests in flight. They do not go through mmap page faults.
- **Compute:** the CPU `MUL_MAT_ID` reads the slots through views, as the GPU cache does for its banks.
- **Memory:** the mapped expert pages are never touched, so Windows does not need to keep them resident.
- **Expected gain (estimate):** the 165 MB/token of misses at 2-3 GB/s takes 55-80 ms instead of about 150 ms. The prompt's 16 GiB takes 6-8 s instead of about 30 s.
- **Cost:** several hundred lines and a Windows I/O path; testable on Linux with threads and `pread`. A cheaper variant is prefetch from several threads (option 10), but Windows may drop prefetch hints at A's free-memory level.

## Sources

- Engine source: `src/llama-context.cpp` (`resolve_fused_ops`), `src/models/glm5-next.cpp` (HC pre/post, DSA selection), `ggml/src/ggml-backend.cpp` (split inputs, `ggml_backend_tensor_copy`), `ggml/src/ggml-rpc/ggml-rpc.cpp` (`cpy_tensor`, `GRAPH_RECOMPUTE`), `ggml/src/ggml-cuda/ggml-cuda.cu` (CUDA graph warmup and reuse).
- [PrefetchVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory) (Microsoft Learn): a strong hint that can completely or partially fail under low-memory conditions.
- [llama.cpp discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324): overlapped reads against mmap paging for MoE experts on Windows.
