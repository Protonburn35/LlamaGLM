# v0.8.4 test plan

v0.8.4 adds the options found in the v0.8.3 test ([TEST-RESULTS-v0.8.3.md](TEST-RESULTS-v0.8.3.md)). As before, everything new is off unless a switch turns it on, and the v0.8.3 switches work unchanged.

## New in v0.8.4

| Where | Switch / plan | What it does | Off by default because |
|---|---|---|---|
| A, B, C | `-LockClocks` | holds the GPU at full clocks while the server or worker runs, and resets the clocks when it exits (`gpu-clocks.ps1`). Memory goes to its highest supported clock and graphics to 75-100% of its highest. On 2026-10-25 all three GPUs stayed in P8/P5 with 405-810 MHz memory while generating. | needs administrator rights; the GPU draws more power while locked |
| A | `-Plan relief-mtp` | `relief` with room for the MTP block: Stage A keeps layers 0-16, C's RAM 34-41, and C's GPU 42-44 + MTP block + output. Use it with `-Mtp`. | needs about 24 GiB free in C's RAM at load, like `relief` |
| B, C | `-ChunkedLoad` | the worker reads cached weights in 64 MiB pieces instead of whole tensors (up to 1 GiB extra RAM during loading before) | new, tested only on the CPU test model |
| A | `-MoeCacheMib` (changed) | warns when the cache is below 4096 MiB: one token needs about 1 GiB of experts, and 1024 MiB gave 0% hits | |
| | `gpu-clocks.ps1` | the same lock by hand: `.\gpu-clocks.ps1` to lock, `.\gpu-clocks.ps1 -Reset` to undo (administrator window) | |

About `-LockClocks`:
- **A:** start PowerShell on A **as administrator**. Otherwise Windows asks for permission when the server starts and again when it stops.
- **Workers:** an SSH session of an administrator account already has the rights, so no prompt appears.
- **Ctrl+C** resets the clocks. If a window is closed instead, run `.\gpu-clocks.ps1 -Reset` as administrator, or reboot.
- **Laptop:** some laptop GPUs refuse to lock their memory clock. The `[GPU CLOCKS]` line then says "not supported" for that part, and the graphics lock still applies.

## Session

All commands below run from A; the workers are started over SSH as before.

Workers, once:
```powershell
ssh -t harym@192.168.50.143 "cd /d C:\LlamaGLM && powershell -NoProfile -ExecutionPolicy Bypass -File .\start-stage-c.ps1 -SkipSelfTest -Timing -LocalCopy -LockClocks -ChunkedLoad"
ssh -t harym@192.168.50.143 "cd /d C:\LlamaGLM && powershell -NoProfile -ExecutionPolicy Bypass -File .\perf-log.ps1 -Role C"
ssh -t harym@192.168.50.129 "cd /d C:\LlamaGLM && powershell -NoProfile -ExecutionPolicy Bypass -File .\start-stage-b.ps1 -SkipSelfTest -Timing -LocalCopy -LockClocks -ChunkedLoad"
ssh -t harym@192.168.50.129 "cd /d C:\LlamaGLM && powershell -NoProfile -ExecutionPolicy Bypass -File .\perf-log.ps1 -Role B"
```
In each worker window, check the `[GPU CLOCKS]` line says "locked".

Each run on A runs in an administrator window, with A's logger in a second window. Send the same prompt each time, let it generate for about 3 minutes, and stop the answer in the client. Then Ctrl+C the logger and the server.
```powershell
cd C:\LlamaGLM
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-stage-a.ps1 -SkipSelfTest -Timing -Prefetch -FusedHc -LockClocks <plan switches>
powershell -NoProfile -ExecutionPolicy Bypass -File .\perf-log.ps1 -Role A
```

| Run | Plan switches | Compare with |
|---|---|---|
| 1 | `-Plan relief` | v0.8.3 run 3: 435 ms/token, 2.46 tok/s |
| 2 | `-Plan relief-mtp -Mtp` | v0.8.3 run 5: 2.79 tok/s |
| 3 | `-Plan gpu-attn` (no `-MoeCacheMib`) | run 1: does moving A's non-expert work to a 4090 at full clocks help |
| 4 (optional) | (none: `balanced`) | v0.8.3 run 2: 462 ms/token |

Afterwards, stop the workers with Ctrl+C (this resets their clocks), then collect the logs:
```powershell
ssh -t harym@192.168.50.143 "cd /d C:\LlamaGLM && powershell -NoProfile -ExecutionPolicy Bypass -File .\collect-logs.ps1"
ssh -t harym@192.168.50.129 "cd /d C:\LlamaGLM && powershell -NoProfile -ExecutionPolicy Bypass -File .\collect-logs.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File .\collect-logs.ps1
```
Copy the worker zips to A with `scp harym@<ip>:C:/LlamaGLM/<zip name> C:\LlamaGLM\`.

## What to look at

- **Clocks:** `gpu-*.csv` should show P0/P2 with the memory at its full clock while generating.
- **Speed:** `sched_profile` lines on A. The 4090's `run`, RPC0's `fetch` (laptop GPU) and `rest` (C's GPU) should drop compared with v0.8.3.
- **MTP:** `draft acceptance`, when a request finishes on its own.

## Rollback

Leave the switches off. If clocks stay locked after a closed window: `.\gpu-clocks.ps1 -Reset` as administrator, or reboot.
