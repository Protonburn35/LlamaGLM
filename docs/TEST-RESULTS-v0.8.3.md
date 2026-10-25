# v0.8.3 test results (2026-10-25)

Inputs: the user's logs from the three PCs (HAR = A, PROART = B, ASUSPC = C).
- Six completed runs between 08:13 and 09:06, plus two failed starts at 07:41 and 07:50.
- Workers ran with `-Timing -LocalCopy` from 08:10, the later ones started over SSH.
- Every run used the same 215-token prompt and was stopped after a few minutes, except run 3, which generated 1074 tokens.
- A's GPU log is missing for run 4.

"ms/token" below is the median of the 30-s `sched_profile` windows with at least 20 tokens. "tok/s" is the median of the server's 3-s `tg_3s` samples. Numbers marked "estimate" are arithmetic, not measurements.

## 1. Runs

| Run | A switches (all with `-Timing -Prefetch`) | ms/token | tok/s | Notes |
|---|---|---|---|---|
| 2026-10-25 earlier | v0.8.2, no local copies | 549 | 1.98 | from [SPEED_RESEARCH-2026-10-25.md](SPEED_RESEARCH-2026-10-25.md) |
| 1 | balanced | 549 | 2.18 | local copies on the workers |
| 2 | `-FusedHc` | 462 | 2.30 | graph 14758 to 7738 nodes |
| 3 | `-FusedHc -Plan relief` | 435 (best window 401) | 2.46 | 1074 tokens; A reads half as much from its SSD |
| 4 | `-FusedHc -Plan gpu-attn -MoeCacheMib 1024` | 699 | 1.47 | the expert cache never hit (section 2.5) |
| 5 | `-FusedHc -Plan tight-mtp -Mtp` | 742 per 2-token step | **2.79** | best output speed |
| 6 | `-FusedHc -NoCudaGraphs` | 515 | 2.30 | A's paging was higher in this run; the 4090 took about 5 ms more |

Prompt processing: 215 tokens took 36-42 s (5.1-6.0 tok/s), against 58.9 s (3.65 tok/s) earlier. A's CPU split is 25-31 s of that; the workers' share fell from about 6-6.7 s to about 2 s.

## 2. Findings

### 2.1 All three GPUs stay in power-saving states while generating

The `gpu-*.csv` logs show the GPUs reaching full clocks only during the prompt. While generating tokens they stay at:

| GPU | pstate while generating | SM clock | memory clock | memory clock at full speed |
|---|---|---|---|---|
| A RTX 4090 | P8 | 210-465 MHz | 405 MHz | 11351 MHz |
| B RTX 4060 Laptop | P5 (P4 at times) | 210-300 MHz | 810 MHz (6001 at times) | 8401 MHz |
| C RTX 5070 Ti | P5 / P8 | about 950 MHz average | 405-810 MHz | 13801 MHz |

This explains why GPU layers were slower per layer than CPU layers.
- **The laptop GPU** took 10.7 ms per token for its two layers in the window where its memory ran at 6001 MHz (run 2, P4). At 810 MHz it took 38-39 ms (run 3 and others).
- **C's GPU** took 38-42 ms in run 1's windows with memory up to 7001 MHz, and about 50 ms at 810 MHz.
- **The 4090** stayed at P8 with 405 MHz memory for the whole of every run and took 55-64 ms for 7 layers. Neither halving the operations (run 2) nor turning off CUDA graphs (run 6) changed that much.

The GPUs are busy only about 10-13% of each token, so the driver keeps them in low-power states. NVIDIA Control Panel's power management mode applies to 3D applications and evidently does not cover these CUDA processes. `nvidia-smi --lock-gpu-clocks` and `--lock-memory-clocks` from an administrator console hold the clocks up until reset or reboot; not every GPU accepts them ([Zivid](https://support.zivid.com/support/unstable-capture-time-windows.html), [NVIDIA forum](https://forums.developer.nvidia.com/t/can-not-to-lock-gpu-clock-rtx-4090/286603)).

Estimate at full clocks: the 4090's 7-8 layers drop from 58-71 ms to about 10-15 ms. The laptop GPU drops to about 5-10 ms. C's GPU and output head drop from about 50 ms to 10-15 ms. Together that is roughly 80-110 ms per token.

### 2.2 The fused HC post works

`-FusedHc` keeps the fused op on ("which both support it (LLAMA_FUSED_HC_FIX)"). It halves the graph to 7738 nodes and cuts each boundary to 2 hidden-state copies, so the workers now send 128 KiB per hop instead of 384 KiB.
- **Result:** 549 to 462 ms/token. B RAM fetch went from 49.5 to 42.3 ms, B GPU from 52.8 to 14.2 ms, C RAM from 48.5 to 38.4 ms.
- **Compute did not get faster with half the nodes:** worker CPU layers stay at about 5.5 ms each, and GPU time is set by the clocks in 2.1.

### 2.3 Local copies work

With `-LocalCopy`, B's and C's RAM devices send 0 KiB per token; their outputs go to their own GPUs on the worker. Against the earlier session, B RAM and C RAM fetch times fell by about 13 ms each (run 1).

### 2.4 `relief` halves A's paging

- **SSD reads:** with two layers fewer on A, its SSD reads fell from 322 to 172 MB/s (about 150 to 75 MB per token) at a read queue of 0.26 instead of 0.59.
- **Time:** A's CPU split fell from 245 to 176 ms (best window 145 ms). The extra layers on the 4090 and C's RAM cost about 13 and 6 ms.
- **Stalls:** about 8% of samples below 1.3 tok/s, against 19% earlier. The slowest token of each window still spends 1-4 s in A's CPU split.
- **Memory headroom:** `relief` uses nearly all of it: the 4090 had about 370 MiB above its margin, and C's RAM about 1.6 GiB.

### 2.5 The MoE cache thrashes at this size

In `gpu-attn`, `-MoeCacheMib 1024` gave 873 MiB of cache for 40 GiB of host experts, with 0 hits in 23385 lookups and 200 GiB uploaded over PCIe. One token needs 8 experts in each of 16 layers (about 1 GiB). A cache smaller than that evicts everything every token, and the uploads (CPU fetch 341-559 ms) made run 4 the slowest. `gpu-attn` itself is untested without the cache. In run 4 the 4090 ran the non-expert parts of 25 layers in 67-100 ms at P8 clocks.

### 2.6 MTP gives the highest output rate

With `tight-mtp -Mtp`, a 2-token verification step took a median 742 ms, and output reached 2.79 tok/s (median of samples; 2.58 mean). The server log has no acceptance count because the request was stopped. The step's CPU split was 427 ms, against 245 for one token, because two tokens select more experts that A must page in. A `relief`-style MTP plan does not exist yet: `tight-mtp` keeps 18 layers on A.

### 2.7 The failed starts were the laptop running out of memory during loading

Both failed starts (07:41, 07:50) stopped while the laptop loaded its RAM layers from its weight cache, with nothing logged afterwards. That pattern fits a failed allocation: each cached tensor is read whole into memory first, up to 1 GiB. The laptop's page file was about 5 GB then. After it was raised to 26 GB, all six runs loaded. The laptop also had more free RAM after the reboot: 22.6-23.1 GiB at fit time, against 21.4 GiB at 07:35.

## 3. Proposed v0.8.4 (opt-in unless noted)

1. **`-LockClocks` on A, B and C.** From an elevated window, lock the GPU's memory clock to its highest supported value and its graphics clock to a high minimum (`nvidia-smi --lock-memory-clocks`, `--lock-gpu-clocks`, values read from `--query-supported-clocks`). Reset them when the launcher exits. `perf-log.ps1` already shows whether it held. Expected: about 80-110 ms/token (estimate).
2. **Plan `relief-mtp`:** `ngl 30`, `ts 8,7,2,8,5`. A keeps 0-16, C's RAM 34-41, and C's GPU 42-44 + MTP block + output.
3. **Worker:** load cached weights in pieces instead of whole tensors (removes the 1 GiB load-time spike behind 2.7).
4. **MoE cache:** warn when `-MoeCacheMib` cannot hold one token's experts, and drop it from the test plan; retest `gpu-attn` without it once clocks are locked.
5. **Defaults (for the user to decide):** `-FusedHc` and `-LocalCopy` gave identical results and only gains; they could become defaults with switches to turn them off.

v0.8.4 implements 1-4 as options (`-LockClocks`, `relief-mtp`, `-ChunkedLoad`, the cache warning); 5 stays the user's choice. See [TEST-PLAN-v0.8.4.md](TEST-PLAN-v0.8.4.md).

Expected with 1-2 (estimate): `relief` at about 330-350 ms/token (about 3 tok/s) and MTP at about 3.5-4 tok/s. A's paging, about 75-150 MB per token, then remains the largest cost: option 6 of [SPEED_RESEARCH-2026-10-25.md](SPEED_RESEARCH-2026-10-25.md).
