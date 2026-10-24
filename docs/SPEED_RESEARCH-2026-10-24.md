# Speed research (2026-10-24)

Research only: no code or configuration was changed. Inputs: the user's `Llamaglm logs.zip` (v0.8.1, commit `bb0bc7d`), [HARDWARE_HANDOFF-2026-10-24.md](HARDWARE_HANDOFF-2026-10-24.md), the engine source and the public sources at the end. Constraints kept: same model and mixed quantization, all 8 selected experts, normal nonzero-temperature sampling, reasoning on, no silent context reduction. Every number marked "estimate" is arithmetic, not a measurement.

## 1. What the logs measure

| Item | Value | Source |
|---|---|---|
| Decode, MTP off, `ngl=29` | 586 ms/token, 1.71 tok/s over 518 tokens | `llama-server-20261024-134750.log` |
| Prefill, same run | 214 tokens in 62.6 s (3.42 tok/s); the first 203-token ubatch returned after 49.3 s and the pipeline drained about 7 s later, at the first checkpoint | same |
| Decode, MTP `n_max=1`, `ngl=28` | 227 verification steps of 949 ms each, 1.81 tokens per step, 524 ms/token (1.90 tok/s) | `llama-server-20261024-141606.log` |
| MTP drafting | 4.3 s for 227 drafts (19 ms each, about 2% of generation time) | same |
| Graph splits / reallocations | 6 splits, 0 reallocations | both |
| Network per token | about 1.8 MiB (B CPU 0.38, B GPU 0.38, C CPU 0.38, C GPU 0.62 MiB including logits); 25 Mbps at 1.7 tok/s, about 15 ms/token of 1 GbE time | worker `[stats]` deltas |
| A memory | 31.9 GiB RAM, 22.5 GiB free before load; **41.2 GiB** of weights memory-mapped from the SSD (43.8 GiB in the MTP plan) | server memory breakdown |
| Free after load | A GPU 4.1, B RAM 4.8, B GPU 0.56, C RAM 5.6, C GPU 3.15 GiB | baseline shutdown breakdown |
| Decode stalls | every 15-30 s a 3-s window drops to 0.6-1.2 tok/s | `tg_3s` lines |

## 2. Where the time goes

### Decode

The baseline and MTP runs give a rough cost model: a step with n tokens costs about F + n*V. With T(1) = 586 ms and T(2) = 949 ms, V is about 364 ms and F about 221 ms (estimate; the MTP run had one more CPU layer on A).

V is the work that grows with the number of tokens, which in decode is mostly the routed experts each token selects: 68.5 MiB per MoE layer, about 1.03 GiB per token in A's CPU layers 3-17 and 0.47 GiB each in B's and C's RAM layers. Read from RAM at dual-channel DDR5 speed (about 60 GB/s), that is roughly 35-50 ms, not 364 ms. The effective rate is about 5 GB/s.

The most likely cause is **A paging experts from its SSD**. A maps 41.2 GiB of weights with at most about 20 GiB of page cache, so roughly half of its 36 GiB of routed experts cannot stay resident, and every token touches some that are not. Windows serves those through page faults of about 32 KiB each ([Raymond Chen](https://devblogs.microsoft.com/oldnewthing/20160909-00/?p=94285%2F)). In the upstream disk-offload discussion, a Windows/CUDA user measured 377 MB/s for MoE experts through mmap paging against 2.80 GB/s with 16 MiB overlapped reads ([llama.cpp discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324)). At a few hundred MB/s, 130-350 MiB of non-resident experts per token explains V. It also explains the user's observation that the SSD was "not hit hard": small dependent faults leave a fast NVMe mostly idle while the CPU waits.

F (about 221 ms) covers the non-routed weights computed on CPUs (about 2.6 GiB/token on A, 0.9 GiB each on B and C), many small hyper-connection ops, the RPC round trips and about 15 ms of network transfer. It is larger than the bandwidth estimate (about 110 ms), so CPU kernel efficiency or extra faults on A may also contribute. Only profiling can split it.

### Prefill

A 203-token ubatch selects 1,624 experts per layer and touches about 287 of the 288 experts in every MoE layer. A therefore reads its whole 41 GiB mapped set per ubatch, and at least 19 GiB of it cannot be resident: at 0.4-1 GB/s that alone is 20-50 s of the measured 56 s. CPU compute of 32 CPU-resident layers (about 8 TFLOP for 203 tokens, estimate) is the other part. Prefill is not limited by the network.

### What it is not

- **Network bandwidth:** about 15 ms of 586 ms per token.
- **RPC latency:** 6 splits and a few blocking round trips per token. `TCP_NODELAY` is set on both ends (`ggml/src/ggml-rpc/transport.cpp`), so Windows delayed-ACK stalls do not apply.
- **Graph rebuilds:** zero reallocations.
- **MTP drafting:** 2% of time.
- **The laptop's bandwidth alone:** B's 7 RAM layers carry 0.47 GiB of routed experts per token, a small share of 586 ms unless B is paging (check below).

### Why MTP helped little

MTP accepted 81% of drafts, but a 2-token verification costs 1.62x a 1-token step because the second token selects different experts, which doubles the expensive paging part. Gain: 1.81 / 1.62 = 1.11x. MTP's value grows once per-token expert cost (V) falls: with V at 100 ms instead of 364 ms, the same acceptance would give about 1.4x (estimate). llama.cpp verifies drafts by sampling from the target model at every position and accepting a draft token only when it matches, so the output distribution is the target's at any temperature.

### Supporting evidence

- The same-model Strata run on A and C without the laptop has even less RAM (about 88 GiB of RAM + VRAM for a 109.5 GiB model), so it also has to read experts from SSD, and it was faster. That fits "how experts are read" being the main lever, not node count or the network.
- One GLM-5.3-Flash user reported the upstream GPU expert cache at 43.5% hits with about 10% of experts cached, and generation dropped from about 33 to 16 tok/s ([summary of #29887 / #30112 reports](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)). GLM's routing is flatter than the models the cache was tuned on.

## 3. Confirm in the next session (current build, no code)

1. **A, during generation**, in a second PowerShell window:
   ```powershell
   cd C:\LlamaGLM
   typeperf "\Process(llama-server)\Page Faults/sec" "\Memory\Page Reads/sec" "\Memory\Pages Input/sec" "\PhysicalDisk(_Total)\Disk Read Bytes/sec" "\PhysicalDisk(_Total)\Avg. Disk Read Queue Length" "\Processor(_Total)\% Processor Time" -si 1 -o logs\perf-a.csv
   ```
   Paging-bound looks like: Pages Input/sec in the tens of thousands, Disk Read Bytes/sec in the hundreds of MB, queue length near 1-2 and CPU well below 100%. Ctrl+C stops it.
2. **B and C, during generation:** `typeperf "\Processor(_Total)\% Processor Time" "\Memory\Available MBytes" "\Memory\Pages Input/sec" -si 1 -o logs\perf-b.csv` (perf-c on C). Pages Input/sec above zero would mean worker weights are being paged, which would be a separate large slowdown.
3. **Fault concurrency on A:** repeat one request with `$env:LLAMA_ARG_THREADS = 16` set before `start-stage-a.ps1` (more threads issue more faults at once). Compare tok/s.
4. **Multi-turn:** send a follow-up message in the same chat and read the server log for `cached n_tokens` and `restored context checkpoint`. If the second turn reprocesses the previous answer, every follow-up costs minutes at 3.4 tok/s; that is a separate, fixable cost.

Use the same prompt and placement for every comparison and record tok/s for prefill and decode separately.

## 4. Options

| # | Option | Kind | Expected effect (estimate) | Effort / risk |
|---|---|---|---|---|
| 1 | Profiling: per-device split time on A, busy time per device in worker `[stats]` | small patch, opt-in | none by itself; tells which of the items below matter | low / low |
| 2 | **Expert prefetch on A**: after the router selects experts for a CPU-resident layer, ask Windows to read those experts' byte ranges with `PrefetchVirtualMemory` (large concurrent reads) instead of waiting on 32 KiB faults; for large batches, prefetch the next layer's expert tensors while the current one computes | small patch in the CPU MoE path, opt-in | if paging is confirmed: decode V from about 364 ms to about 100-150 ms (about 2.5-3 tok/s); prefill paging part 2-5x faster | low-medium / low (a hint; results unchanged) |
| 3 | **Fit more of A's layers into the other PCs' measured slack** (shipped `balanced` plan `ngl=31`, `ts=7,8,2,8,6`) by lowering the Gate 2 RAM margins for B and C, keeping the worker reserves | config only | A maps about 5 GiB less and has 2 fewer CPU layers: 10-25% faster decode if paging-bound | none / medium on B (laptop RAM about 2 GiB free after load; watch Pages Input/sec) |
| 4 | Free RAM on A (close other programs; 9.4 GiB was in use before load) and keep the model on A's fastest NVMe | operational | each freed GiB is one more GiB of page cache | none / none |
| 5 | MTP as an opt-in launcher option, measured on one fixed placement (`n_max` 1 and 2) | launcher | about 1.1x now; larger once option 2 or 9 lowers V | low / low |
| 6 | Worker-local RAM-to-GPU copies in the RPC server (today a worker's CPU-to-CUDA copy fails and goes through A) | small server patch, protocol unchanged | saves a few ms per token now; required before splitting layers within a worker | low / low |
| 7 | Op offload for prefill on A (`-OpOffload`, existing) with a larger ubatch | config only | uncertain: the GPU computes faster, but its copies from mapped pages also fault; test after option 2 | none / Gate 2 guards VRAM |
| 8 | Explicit async expert reader on A (large overlapped reads into a RAM expert cache, as in #23324 and Strata) | larger engine work | the 2.8 GB/s path; only if option 2 falls short | high / medium |
| 9 | **A: 64 GB RAM** (2 x 32 GB DDR5) | hardware | A's mapped weights fit in RAM: paging disappears; decode about 3-4 tok/s before other changes, prefill much faster | none / none |
| 10 | Worker-side GPU offload for large batches (worker GPU computes its RAM layers during prefill) | medium engine work | prefill on B/C layers several times faster for long prompts | medium / medium |
| 11 | Upstream GPU expert cache (`--moe-cache-mib`) on the 4090 for A's layers | config + placement | low expectation for GLM (43.5% hits reported at about 10% cache, with a slowdown); A's limit is RAM capacity, not GPU compute | low / medium |
| 12 | AVX-512 CPU backend variants for Zen 4/5 | build + launcher | small, mainly prefill CPU compute | low / low |

Minor: the boundary between A's last CPU layer and the 4090 disables the fused hyper-connection "post" op for every layer (`resolve_fused_ops` warning); worth a look after the items above.

## 5. Proposed next build (everything off by default)

1. Profiling summaries (option 1).
2. Expert prefetch on A (option 2), with a switch and a counter of prefetched bytes.
3. Worker-local copies (option 6).
4. Launcher options: MTP on/off and `n_max`, thread overrides, per-plan Gate 2 margins, op offload and ubatch; the projection and the server must use the same effective settings, including MTP. RPC test stays on; the CUDA self-test can be skipped on repeat runs.

Test matrix, same prompt and placement each time: baseline; prefetch on; balanced-tight placement; both; each of those with MTP `n_max=1`. Record prefill tok/s, decode tok/s, A's perf counters, and peak RAM/VRAM per device.

## 6. Not recommended now

- **Removing the laptop:** A would have to page roughly another 20 GiB.
- **Faster networking:** about 15 ms/token today.
- **Tensor parallelism:** it does not address A's paging.
- **The stage-runtime branch as the near-term path:** it is not production-ready, although its expert-cache and storage components are relevant to option 8.

## Sources

- [PrefetchVirtualMemory (Microsoft Learn)](https://learn.microsoft.com/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory)
- [Raymond Chen: reads around a faulting page](https://devblogs.microsoft.com/oldnewthing/20160909-00/?p=94285%2F)
- [llama.cpp discussion #23324: MoE offload to disk with on-demand paging](https://github.com/ggml-org/llama.cpp/discussions/23324)
- [llama.cpp MoE offload: --n-cpu-moe vs the new expert cache](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)
