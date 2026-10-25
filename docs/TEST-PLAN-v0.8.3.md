# v0.8.3 test plan

v0.8.3 adds the fixes and experiments from [SPEED_RESEARCH-2026-10-25.md](SPEED_RESEARCH-2026-10-25.md). As before, everything new is off unless a switch turns it on: `.\start-stage-a.ps1` and the workers without switches run like v0.8.2 (plan `balanced`, port 8081). The v0.8.2 switches (`-Prefetch`, `-Timing`, `-Mtp`, `-LocalCopy`, ...) work as before; see [TEST-PLAN-v0.8.2.md](TEST-PLAN-v0.8.2.md).

## New in v0.8.3

| Where | Switch / plan | What it does | Off by default because |
|---|---|---|---|
| A | `-FusedHc` | keeps the fused hyper-connection post op on (`LLAMA_FUSED_HC_FIX=1`). Without it, a placement check at the CPU-to-4090 boundary turns the fused op off for the whole model. The fallback doubles the number of GPU/CPU operations and makes every handoff between PCs send 18 tensors instead of 2 (about 90 round trips per token). The fused and unfused ops compute the same thing; on the test model the text is identical. | new, not yet run on the PCs |
| A, B, C | `-Timing` (changed) | the timing lines also show the **slowest token** of each 30-s window and which device it spent the time on (`slowest graph` on A, `slowest` in the worker `[stats]`), to locate the stalls | (timing is opt-in as before) |
| A, B, C | `-NoCudaGraphs` | diagnostic: turns CUDA graphs off on that PC's GPU (`GGML_CUDA_DISABLE_GRAPHS`). If the GPU time does not change, they were not working. Slower if they were. | diagnostic only |
| A | `-Plan relief` | `tight` plus one more layer in C's RAM: Stage A keeps 16 layers instead of 18 (about 5 GiB less to page from its SSD) | C's RAM fits it only with about 24050 MiB (23.5 GiB) free at load (24789 MiB on 2026-10-25); Gate 2 decides |
| A | `-Plan gpu-attn` | the 4090 also runs attention, the shared expert and the hyper-connections of layers 0-18. Their routed experts stay in A's RAM, so the 7800X3D computes only those (`--n-cpu-moe 19`). Workers as in `balanced`. | experimental: more work moves to the 4090 and every layer hands off between CPU and GPU; can be slower until the GPU cost per operation is known |
| A | `-MoeCacheMib N` | with `gpu-attn`: keeps the most used routed experts of A's RAM layers in N MiB of the 4090 (upstream SLRU cache; Gate 2 counts it) | only works with `gpu-attn` |
| A, B, C | `perf-log.ps1` (changed) | also writes `logs\gpu-<role>-*.csv`: power state, SM and memory clocks, utilization, power, temperature four times per second | (logging only) |

Also fixed: with `-Timing` (v0.8.2), the last timing report on shutdown read device names after the devices were freed. Locally it crashed in about half of the runs; on Windows it can show up as a crash after Ctrl+C. Generation was not affected.

## Before the runs (once)

1. Install v0.8.3 on all three PCs (same zip everywhere). Run the first start of each PC without `-SkipSelfTest`.
2. On all three PCs: NVIDIA Control Panel > Manage 3D settings > Program Settings > Add:
   - A: `C:\LlamaGLM\llama-server.exe`
   - B and C: `C:\LlamaGLM\ggml-rpc-server.exe`
   - Set **Power management mode = Prefer maximum performance** for it. The existing "CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback" stays.
3. Laptop: plugged in, Windows power mode "Best performance", and the ASUS performance or turbo profile.
4. Close browsers and other large programs on A: every GiB free on A is a GiB less read from its SSD.

## Session

Workers (C, then B), with timing and local copies for the whole session:

```powershell
cd C:\LlamaGLM
.\start-stage-c.ps1 -Timing -LocalCopy     # on B: .\start-stage-b.ps1 -Timing -LocalCopy
# second window on each worker:
.\perf-log.ps1 -Role C                      # on B: .\perf-log.ps1 -Role B
```

Each run on A: start the server, start the logger in a second window, send **the same prompt** each time, and let it generate for **about 3 minutes** (enough for 6 timing windows), then stop the answer in the client. Then Ctrl+C the logger and the server before the next run.

```powershell
cd C:\LlamaGLM
.\start-stage-a.ps1 -SkipSelfTest -Timing -Prefetch <switches for the run>
# second window, once the server says it is listening:
.\perf-log.ps1 -Role A
```

Runs, most important first. If time runs out, stop after any run.

| Run | Switches on A (besides `-SkipSelfTest -Timing -Prefetch`) | Question it answers |
|---|---|---|
| 1 | (none) | new baseline: v0.8.2's run with local copies on the workers and GPUs at maximum performance |
| 2 | `-FusedHc` | the main fix: fewer operations and round trips |
| 3 | `-FusedHc -Plan relief` (if Gate 2 refuses it: `-Plan tight`) | less paging on A |
| 4 | `-FusedHc -Plan gpu-attn -MoeCacheMib 1024` (the 4090 has an estimated 1-1.5 GiB to spare in this plan; if Gate 2 refuses it, try 512, or drop `-MoeCacheMib`) | does moving A's non-expert work to the 4090 help |
| 5 | best of runs 2-4, with its `-mtp` plan if one exists (`-Plan balanced-mtp` or `tight-mtp`) and `-Mtp` | MTP on top |
| 6 (optional) | run 2 again with `-NoCudaGraphs` | were CUDA graphs working on the 4090 |

Afterwards, on each PC: `.\collect-logs.ps1`, and send the zips. They include the `perf-*.csv` and `gpu-*.csv` files.

## What to record per run

From the server log: the `[OPTIONS]` line, `fused DeepSeek V4 HC post enabled` (runs 2-5), `eval time` (tokens per second), `draft acceptance` with MTP, and the `sched_profile` lines including `slowest graph`. From the worker windows: the last `[stats]` line of each device. If a gate fails, note which one; Gate 2 and the worker reserves refuse plans that do not fit.

## Rollback

Every experiment is a switch: leave it off. Without switches, `.\start-stage-a.ps1` and the workers run like v0.8.2.
