# Tensor and state ledger

The machine-readable inventory is `docs/models/glm53-ud-iq3xxs.json`, generated from all 1,412 tensor descriptors at Unsloth revision `a38483c8cd5df544f53d70fb281afe97369d5ab6`. Four files total 120,367,571,840 bytes (120.37 GB decimal). No tensor payload is retained in the development environment. The first shard contains metadata and zero tensors; the remaining descriptors occupy 38,304, 35,232 and 15,328 bytes.

`UD-IQ3_XXS` is the publisher's dynamic quant name. Actual tensor types are q6_K (295), f32 (638), q8_0 (350), iq3_s (41), iq2_s (82), iq4_xs (3), q3_K (1), q2_K (2). Substitution with uniformly IQ3_XXS or Q3_K tensors would change the model.

| Stage | Target weights GiB | Routed subset GiB | Target state at 1M GiB | All target weights + state GiB | Optional draft weights GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A [0,22) | 50.368978 | 46.705078 | 8.821075 | 59.190054 | 0 |
| B [22,27) | 12.695821 | 12.041016 | 1.766724 | 14.462544 | 0 |
| C [27,45) | 46.419694 | 43.576172 | 8.804352 | 55.224046 | 2.607678 |

These are tensor payload sizes and persistent numerical-state arrays, not a Windows admission result. The all-resident column is a lower bound for an all-weights-resident placement; desktops may later keep cold experts on local SSD. RAM and VRAM are separate physical budgets. The apparent laptop headroom does not prove residency, working-set locking, WDDM availability or allocation success.

Per DSA layer, FP16 latent MLA requires `1048576 * 512 * 2 = 1 GiB`; index rows require `1048576 * 128 * 3 * 2 = 0.75 GiB`. Eleven trunk DSA layers therefore need 19.25 GiB total, split 8.75 / 1.75 / 8.75 GiB. BF16 rows have the same byte size. One optional DSA NextN context would add another 1.75 GiB at 1M before metadata/workspace; it is not enabled.

Source `llama_hparams::n_embd_r/n_embd_s` gives each KDA layer `3*(4-1)*64*128 = 73728` convolution values and `64*128*128 = 1048576` recurrent values. `llama_model::create_memory` uses FP32 for both. One sequence with no speculative rollback needs 4,489,216 bytes/layer. The 17 / 4 / 13 KDA split contributes the remaining state in the table. Speculative recurrent snapshots multiply those arrays by `1 + n_rs_seq`; they are not free and are not yet planned into runtime admission.

Still to measure/account: host and device mirror copies, Windows/display reserves, actual GPU budgets, cache-cell metadata and pool maps, allocator alignment, CUDA graphs/scratch, prefill high water, expert cache leases, pinned staging, network queues, vision, prefix checkpoints and bounded snapshot buffers. The planner reports `admission=NOT_VALIDATED`. It cannot start inference or claim a 1M allocation succeeded.

Regenerate without weights:

```sh
python3 scripts/glm-fetch-metadata.py /path/to/new-headers
build/bin/llama-glm-plan --output memory.json /path/to/new-headers/*.header
```

Use headers in lexical shard order. Explicit `--split 21,6,18` or neighboring complete splits can be compared; defaults never shrink the context. B must preload every assigned expert, establish resident RAM/VRAM storage, warm/touch required state and fail closed on unestablished residency or later pressure. Mmap alone is insufficient. No runtime provider implementing this guarantee exists yet.
