# LlamaGLM first run (v0.8.2)

No inference has worked on these PCs yet. This version runs cheap checks first and stops at the first problem, so a bad plan or a broken GPU kernel shows up as a clear message instead of a frozen desktop. Do the steps in order; each one stops on failure.

Expect it to be slow: estimated 1.5-4 tokens/s output and 15-50 tokens/s prompt processing (not measured yet). See [`LIMITATIONS.md`](LIMITATIONS.md) for why, and what the hardware could do at best.

| PC | Role | Address | Window you keep open |
|---|---|---|---|
| A | RTX 4090, coordinator + API `http://127.0.0.1:8081` | 192.168.50.206 | `start-stage-a.ps1` (and `smoke-test.ps1`) |
| B | RTX 4060 Laptop worker, port 50052 | 192.168.50.129 | `start-stage-b.ps1` |
| C | RTX 5070 Ti worker, port 50053 | 192.168.50.143 | `start-stage-c.ps1` |

Only PC A needs the model (`C:\GLM-5.3-Flash\UD-IQ3_XXS\`, 4 shards, 112.10 GiB). The workers receive their layers over the LAN and keep a copy in `C:\LlamaGLM-Cache` (B about 26 GiB, C about 32 GiB of disk), so later loads skip the transfer.

## 0. One-time setup (all 3 PCs)

1. Unzip `LlamaGLM-v0.8.2-win-cuda-cu12.8-sm89-sm120-x64.zip` (the download of the "LlamaGLM Windows package" GitHub Actions run) so that `C:\LlamaGLM\llama-server.exe` exists. Use the **same version on all 3 PCs**. Delete older versions (v0.7.0, v0.8.0, v0.8.1) first. To check which build you have: `MANIFEST.txt` starts with `LlamaGLM v0.8.2` and names the source commit, and `.\llama-cli.exe --version` prints the same commit. The "version" number that `llama-cli` prints is llama.cpp's own library version, not the LlamaGLM version.
2. NVIDIA driver R570 or newer (any driver that runs the RTX 5070 Ti is new enough).
3. **NVIDIA Control Panel > Manage 3D settings > Global settings > "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback".** With the default, a full GPU silently spills into shared system RAM, which is one likely cause of the earlier near-freeze.
4. Firewall, Admin PowerShell, once (only PC A may connect):
   - on B: `New-NetFirewallRule -DisplayName LlamaGLM-RPC-50052 -Direction Inbound -Protocol TCP -LocalPort 50052 -RemoteAddress 192.168.50.206 -Action Allow`
   - on C: `New-NetFirewallRule -DisplayName LlamaGLM-RPC-50053 -Direction Inbound -Protocol TCP -LocalPort 50053 -RemoteAddress 192.168.50.206 -Action Allow`
   - if an older rule without `-RemoteAddress` exists: `Remove-NetFirewallRule -DisplayName LlamaGLM-RPC-50052` (or `-50053`) first.
5. If PowerShell refuses to run scripts: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and `Get-ChildItem C:\LlamaGLM\*.ps1 | Unblock-File`.
6. Close large programs on B and C. The default plan uses about 21 GiB of RAM on each worker. Keep them closed while the model runs: a game or many browser tabs on B or C can push worker weights into the page file (very slow) or crash the worker.
7. Laptop (B): plug in the charger, power mode "Best performance", "When I close the lid" = "Do nothing", sleep = "Never", and use a LAN cable if possible. If the laptop sleeps or its Wi-Fi drops, `llama-server` on A stops and must be started again.

If an address changes (DHCP), edit `stage_b.host` / `stage_c.host` in `C:\LlamaGLM\cluster_config.json` on **all** PCs, or reserve the addresses in the router.

## 1. Start the workers (C, then B)

```powershell
cd C:\LlamaGLM
.\start-stage-c.ps1      # on C;  .\start-stage-b.ps1 on B
```

Each worker first runs a **CUDA self-test** on its own GPU (1-5 min, tiny tensors, the exact ops and weight types of GLM-5.3-Flash). If it prints `[FAIL]`, stop there and send `C:\LlamaGLM\logs\stage-*.log`. If it passes, the worker starts and waits. Leave the window open. Later, `[stats]` lines appear showing how much data the worker received and how many graphs it computed.

## 2. Check the plan without loading the model (on A)

```powershell
cd C:\LlamaGLM
.\start-stage-a.ps1 -PlanOnly
```

- Gate 0: files, driver, 4 shards with the right size.
- Gate 1: CUDA self-test on the 4090, then `llama-rpc-test`: each worker must report 2 devices (its GPU and its RAM, with plausible sizes) and compute small quantized matrix products that match the CPU.
- Gate 2: the projected memory of the **real** model on every device, without allocating anything:
  ```
  common_fit_check: CUDA0 (NVIDIA GeForce RTX 4090)   22xxx MiB free, needs 19xxx MiB (model 18200 + ...) + margin 1536 -> ok
  common_fit_check: RPC1 (192.168.50.129:50052)        ...                     (model 20801 + ...) + margin 2560 -> ok
  ...
  ```
  If a line says `DOES NOT FIT`, close programs on that PC and repeat, or move one layer from that device to the Stage A RAM range (section 5).

## 3. Load the model (on A)

```powershell
.\start-stage-a.ps1
```

It repeats Gates 0-2 (quick) and starts `llama-server`. The first load sends about 56 GiB to the workers (about 8-10 min on 1 GbE), then runs a warm-up. Stage A keeps layers 0-15 in RAM through mmap; about 15 GiB of those are read from its SSD while it works, which is normal and does not use the page file. Later loads are faster because the workers read their weight cache.

## 4. Test real generation (second window on A)

```powershell
cd C:\LlamaGLM
.\smoke-test.ps1
```

- Gate 3: waits for `/health` ok, then checks the server log: mmap is on and **all 4 worker devices hold weights** (no silent single-PC fallback).
- Gate 4: a raw completion and a chat completion must return generated tokens (`completion_tokens > 0`) **and the right answers**: "The capital of France is" must continue with Paris, "what is 2 + 2?" must contain 4. Broken GPU kernels still produce tokens, only wrong ones. GLM always thinks first; the test asks for short reasoning and accepts text in `reasoning_content`. If the printed text is clearly correct but worded differently, `.\smoke-test.ps1 -NoAnswerCheck` skips this check.
- Gate 5: two more requests, health still ok, the token counter advances.

Only "Gate 4 PASS" means the model generated text across the three PCs. Expect it to be slow; speed has never been measured.

## 5. Plans and settings

`cluster_config.json` holds everything; the launcher checks that `sum(ts) == ngl`.

| Plan | Stage A RAM (mmap) | 4090 | B RAM | B GPU | C RAM | C GPU |
|---|---|---|---|---|---|---|
| `balanced` (default, ran on 2026-10-24) | layers 0-17, 40.2 GiB | 18-24, 17.8 GiB | 25-31 | 32-33 | 34-40 | 41-44 + output |
| `balanced-mtp` | 0-18, 42.8 GiB | 19-25 | 26-32 | 33-34 | 35-41 | 42-44 + MTP block 45 + output |
| `tight` | 0-16, 37.7 GiB | 17-24, 20.3 GiB | 25-31 | 32-33 | 34-40 | 41-44 + output |
| `tight-mtp` | 0-17, 40.2 GiB | 18-25, 20.3 GiB | 26-32 | 33-34 | 35-41 | 42-44 + MTP block 45 + output |
| `laptop-light` (not tested) | 0-21, 50.4 GiB | 22-28 | 31 | 29-30 (KDA only) | 32-40 | 41-44 + output |

`tight` leaves about 1.6 GiB of the 4090 free at load; Gate 2 refuses it if a desktop program uses more VRAM. MTP needs a plan whose name ends in `-mtp` (room for the MTP block on C's GPU).

Experiments (all off by default; a switch turns one on for one run, or set it in the `options` block of `cluster_config.json`):

| Switch | On | Effect |
|---|---|---|
| `-Prefetch` | A | Windows reads the experts each token selects from Stage A's RAM layers in large parallel reads instead of small page faults; results unchanged |
| `-Timing` | A, B, C | per-device milliseconds per token in the logs (`sched_profile` lines on A, `busy` in the worker `[stats]`) |
| `-Mtp`, `-MtpDraftMax N` | A | speculative decoding with the model's MTP block (draft length 1 by default); the output distribution is unchanged |
| `-Threads N`, `-UBatch N`, `-OpOffload` | A | CPU threads on A, prompt batch size, GPU compute for Stage A's RAM layers during prompts |
| `-LocalCopy` | B, C | copies between the worker's RAM and GPU on the worker instead of through Stage A |

`.\perf-log.ps1 -Role A` (or `-Role B`, `-Role C`) in a second window logs SSD reads, page faults and CPU use once per second to `logs\perf-*.csv`. The test sequence for v0.8.2 is in [`docs/TEST-PLAN-v0.8.2.md`](docs/TEST-PLAN-v0.8.2.md).

To move one layer from a worker's RAM to Stage A, lower that device's `ts` value by 1 and `ngl` by 1 (devices are listed in `devices`, in the same order as `ts`).

## 6. If something fails

- **A window shows an error and exits:** run `.\collect-logs.ps1` on that PC. Review the zip for private data before sharing it.
- **A worker crashed:** it restarts itself up to 3 times. Then start `start-stage-a.ps1` again; the workers do not need to be touched for a Stage A restart.
- **Suspected GPU kernel problem:** start the affected PC with `-Diagnostic` (for example `.\start-stage-c.ps1 -Diagnostic` and `.\start-stage-a.ps1 -Diagnostic`). CUDA then reports the failing operation by name. It is slow; use it only to find the error.
- **Toolchain baseline (official llama.cpp build of the same source):** download from `https://github.com/ggml-org/llama.cpp/releases/tag/b11780` both `llama-b11780-bin-win-cuda-13.4-x64.zip` and `cudart-llama-bin-win-cuda-13.4-x64.zip`, unzip both into `C:\llama-b11780` on **all 3 PCs** (needs driver R580 or newer), then run:
  ```powershell
  .\start-stage-c.ps1 -BinDir C:\llama-b11780      # C, and the same with start-stage-b.ps1 on B
  .\start-stage-a.ps1 -BinDir C:\llama-b11780      # A
  ```
  The same plan and gates are used. The upstream build prints the Gate 2 table but cannot refuse a plan that does not fit, so read the `needs` vs `free` numbers yourself. Both builds use nvcc; if the official build works where ours fails, the difference is in our build (CUDA 12.8 instead of 13.4, or the LlamaGLM patches).
- **Optional, saves walking upstairs:** share the worker log folders read-only on the LAN (on B and C, Admin PowerShell: `New-SmbShare -Name LlamaGLM-Logs -Path C:\LlamaGLM\logs -ReadAccess "$env:USERNAME"`). Then `collect-logs.ps1` on A also copies the worker logs.

## 7. Stop

Ctrl+C in the `start-stage-a.ps1` window, then in the worker windows. Workers can stay running between Stage A restarts.
