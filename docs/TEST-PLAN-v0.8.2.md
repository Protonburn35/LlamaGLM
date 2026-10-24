# v0.8.2 test plan

v0.8.2 adds experiments for speed. Every experiment is off unless you turn it on with a switch, so `.\start-stage-a.ps1` with no switches runs like v0.8.1 (plan `balanced`, the split that ran on 2026-10-24, API on port 8081). Background: [SPEED_RESEARCH-2026-10-24.md](SPEED_RESEARCH-2026-10-24.md).

## Switches

| Where | Switch | What it does | Off by default because |
|---|---|---|---|
| A | `-Prefetch` | asks Windows to read the experts each token selects from Stage A's RAM layers in large parallel reads (`LLAMA_MOE_PREFETCH=1`); results are unchanged | new, not yet run on Windows hardware |
| A | `-Timing` | prints per-device milliseconds per token every 30 s in the server log (`sched_profile` lines) | adds a small synchronization per device |
| A | `-Mtp` / `-MtpDraftMax N` | speculative decoding with the model's own MTP block (default draft length 1); output distribution unchanged | needs a plan with room for the MTP block |
| A | `-Plan balanced-mtp`, `-Plan tight`, `-Plan tight-mtp` | other layer splits (see `cluster_config.json`); `tight` puts one more layer on the 4090 | `tight` leaves about 1.6 GiB of the 4090 free |
| A | `-Threads N`, `-UBatch N`, `-OpOffload` | CPU threads on A, prompt batch size, GPU compute for Stage A's RAM layers during prompts | untested on these PCs |
| B, C | `-Timing` | adds `busy ... ms/graph` to the worker `[stats]` lines (`GGML_RPC_SERVER_PROFILE=1`) | adds a synchronization per graph |
| B, C | `-LocalCopy` | copies between the worker's RAM and GPU on the worker instead of through Stage A (`GGML_RPC_SERVER_LOCAL_COPY=1`) | new, not yet run on a GPU |

The same options can be set in the `options` block of `cluster_config.json`. The launchers now remove leftover `LLAMA_ARG_*` environment variables, so use the switches instead of setting those by hand.

## Session

Install v0.8.2 on all three PCs (same zip everywhere). Workers can stay running between Stage A runs. Use `-SkipSelfTest` on repeat runs of the same build; the RPC check (Gate 1) always runs.

1. **Workers** (C, then B), with timing on for the whole session:
   ```powershell
   cd C:\LlamaGLM
   .\start-stage-c.ps1 -Timing        # on B: .\start-stage-b.ps1 -Timing
   ```
2. **Each run on A**: start the server, then in a second window start the counter logger, then send the same prompt (the one from 2026-10-24 is fine) and wait for the answer. Stop the server with Ctrl+C before the next run.
   ```powershell
   cd C:\LlamaGLM
   .\start-stage-a.ps1 -SkipSelfTest -Timing <switches for the run>
   # second window, once the server says it is listening:
   cd C:\LlamaGLM
   .\perf-log.ps1 -Role A
   ```
   Optional, during one run: `.\perf-log.ps1 -Role B` on the laptop.

| Run | Switches on A | Question it answers |
|---|---|---|
| 1 | (none) | baseline with timing; the counter log shows whether A is reading from its SSD during generation |
| 2 | `-Prefetch` | does prefetch speed up generation and prompts |
| 3 | `-Prefetch -Plan tight` | does one more layer on the 4090 add to that |
| 4 | `-Prefetch -Plan tight-mtp` | same placement as run 5 without MTP; skip if Gate 2 refuses tight-mtp and use balanced-mtp for runs 4 and 5 |
| 5 | `-Prefetch -Plan tight-mtp -Mtp` | MTP gain on the same placement |
| 6 | best of the above, workers restarted with `-Timing -LocalCopy` | worker-local copies |

After the best run, send a follow-up message in the same chat and note whether the second answer starts quickly.

3. **Afterwards**, on each PC: `.\collect-logs.ps1` and send the zips (they now include the `perf-*.csv` files).

## What to record per run

From the server log: `prompt eval time` and `eval time` lines (tokens per second), `draft acceptance` with MTP, and the last `sched_profile` lines. From the worker windows: the last `[stats]` line of each device. If a gate fails, note which one; Gate 2 and the worker reserves still refuse plans that do not fit.

## Rollback

Every experiment is a switch: leave it off. To return to the exact v0.8.1 behaviour, run `.\start-stage-a.ps1` and the workers without switches; the default plan is the one that ran on 2026-10-24.
