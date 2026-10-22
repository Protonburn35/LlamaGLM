# Memory plan

All sizes come from the real tensor table of `unsloth/GLM-5.3-Flash-GGUF` `UD-IQ3_XXS` (headers of the 4 shards read with HTTP range requests; 1412 tensors, 112.092 GiB). GiB = 2^30 bytes.

## 1. Model facts (from the GGUF metadata)

- `glm5-next.block_count = 46`: 45 trunk layers (0-44) + 1 NextN/MTP layer (45). `nextn_predict_layers = 1`.
- Layers 0-2 dense (`leading_dense_block_count = 3`), 3-44 MoE: 288 experts, 8 used, 1 shared expert, expert FFN 2048, hidden 4096, 4 hyper-connection streams.
- DSA (attention) layers = `head_count_kv > 0`: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 (and 45). All others are KDA (recurrent).
- Despite the name, the routed experts are mostly IQ2_S (gate, up) + IQ3_S (down); layers 11, 12 and 44 use larger types.

| Tensor group | Size |
|---|---|
| `token_embd` (Q6_K, input layer, always Stage A CPU) | 0.485 GiB |
| `output` (Q6_K) + `output_norm` | 0.485 GiB |
| layers 0-2 (dense) | 0.222 GiB each |
| MoE layers 3-10, 13-43 | 2.539-2.540 GiB each (routed experts 2.408, rest 0.131) |
| layer 11 / 12 / 44 | 3.283 / 2.767 / 2.767 GiB |
| layer 45 (NextN/MTP) | 2.608 GiB, **not loaded** when MTP is off (tensors get `TENSOR_SKIP`; "unused blk.45.*" warnings are expected) |
| **text-only total** | **109.49 GiB** |

## 2. Why the earlier plans could not work

- 24/3/20 on GPUs only: 54.96 GiB on the 4090 (exactly the failed 59,016,437,120-byte buffer), ~7.6 GiB on the 8 GB laptop GPU, ~46.6 GiB on the 16 GB 5070 Ti. The three GPUs have 48 GB in total.
- Gemini's `-ngl 16 -ts 8,3,5`: `-ngl N` offloads the **last** N slots (layers 31-46), so 73 GiB stayed in Stage A RAM, and its RPC change disabled mmap, so all of it was to be read into RAM (32 GB PC).
- Usable memory: about 43 GiB of VRAM plus about 3 x 24 GiB of RAM after the OS, roughly 115 GiB, against 109.5 GiB of weights. The model only fits if **all six pools** are used: each PC's GPU and RAM. Stage A's RAM part is mmapped, so the part that does not fit in its page cache is re-read from its SSD (clean file pages, no page file). Worker RAM holds weights in normal allocations and must never be overcommitted, so it is limited twice (Gate 2 margin and the worker's own reserve).

## 3. Slots and `-ngl` / `-ts`

llama.cpp numbers the slots 0-45 (layers) and 46 (output). `-ngl N` offloads the last N slots; slots before that stay on the CPU of Stage A. `-ts c1,...,cK` with integer counts that **sum to N** gives exactly `c1` slots to the first `--device`, then `c2` to the next, and so on (upstream arithmetic, no override). The launcher checks `sum(ts) == ngl`. Slot 45 holds no weights with MTP off.

RPC device names depend on the order of `--rpc` and of each worker's `-d CUDA0,CPU`: RPC0 = B GPU, RPC1 = B RAM, RPC2 = C GPU, RPC3 = C RAM. Gate 1 checks this by device size.

## 4. Plans

### balanced (default): `-ngl 31 -ts 7,8,2,8,6 --device CUDA0,RPC1,RPC0,RPC3,RPC2`

| Pool | Slots | Weights | DSA layers |
|---|---|---|---|
| A RAM (mmap) | 0-15 | 35.13 GiB incl. token_embd | 3, 7, 11, 15 |
| A RTX 4090 (CUDA0) | 16-22 | 17.77 GiB | 19 |
| B RAM (RPC1) | 23-30 | 20.31 GiB | 23, 27 |
| B RTX 4060 (RPC0) | 31-32 | 5.08 GiB | 31 |
| C RAM (RPC3) | 33-40 | 20.31 GiB | 35, 39 |
| C RTX 5070 Ti (RPC2) | 41-45 + output | 10.87 GiB | 43 |

### laptop-light: `-ngl 25 -ts 7,2,1,9,6 --device CUDA0,RPC0,RPC1,RPC3,RPC2`

| Pool | Slots | Weights |
|---|---|---|
| A RAM (mmap) | 0-21 | 50.37 GiB |
| A RTX 4090 | 22-28 | 17.77 GiB |
| B RTX 4060 | 29-30 | 5.08 GiB (KDA only) |
| B RAM | 31 | 2.54 GiB |
| C RAM | 32-40 | 22.85 GiB |
| C RTX 5070 Ti | 41-45 + output | 10.87 GiB |

The laptop runs 3 layers as preferred, but they do not fit on its GPU alone (3 x 2.54 GiB + compute > 8 GB), and the cost is about 15 GiB more SSD reading on Stage A per pass over the weights. Keeping the laptop GPU strictly KDA-only in the balanced plan would cost 2.5-5 GiB more on Stage A as well; at 8K context a DSA layer's cache is only 10-20 MiB, so it was not worth it.

## 5. Projection with the real model (Gate 2 dry run)

`llama-fit-params --fit-print on` with the balanced plan on the VM (RPC devices backed by CPU, so compute buffers are CPU sizes; CUDA compute buffers will differ):

| Device | model | context (KV/state, 8K) | compute | margin (`-fitt`) |
|---|---|---|---|---|
| A GPU slot | 18200 MiB | 39 MiB | 550 MiB | 1536 |
| B RAM | 20801 | 53 | 718 | 2560 |
| B GPU | 5200 | 18 | 653 | 512 |
| C RAM | 20801 | 53 | 642 | 2560 |
| C GPU | 11130 | 26 | 641 | 1024 |
| Host (A) | 35977 mmapped, not counted | 107 | 247 | 1536 |

Required free memory at Gate 2 therefore is about: 4090 19.9 GiB, laptop GPU 6.2 GiB, laptop RAM 23.6 GiB, C RAM 23.5 GiB, 5070 Ti 12.5 GiB (CPU-sized compute buffers; the real CUDA numbers appear at Gate 2). On top of that each worker refuses any allocation that would leave less than `reserve_mib` free (B: 384 MiB GPU / 2048 MiB RAM, C: 512 / 2048).

`--no-repack` keeps every CPU weight on Stage A in the mmap; otherwise about 2.8 GiB of Q6_K/Q8_0 tensors would be copied into normal RAM (and Gate 2 would not see them).

`--no-op-offload` is set by default so that prompt processing does not copy whole CPU-resident expert tensors (up to 1.3 GiB) into the 4090's compute buffer. `-OpOffload` turns it back on later.

## 6. Not verified yet

- CUDA compute buffer sizes (only the projection on the real PCs shows them; Gate 2 does this before every load).
- Free VRAM with the desktop running (4090, 5070 Ti) and free RAM on the laptop.
- SSD read rate and token speed on Stage A.
- Longer context: KV grows only on DSA layers (MLA cache); 1M context is not tested and not claimed.
