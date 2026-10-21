# Tensor and state ledger

The machine-readable inventory is `docs/models/glm53-ud-iq3xxs.json`, generated from all 1,412 tensor descriptors at Unsloth revision `a38483c8cd5df544f53d70fb281afe97369d5ab6`. Four files total 120,367,571,840 bytes (120.37 GB decimal). No tensor payload is retained in the development environment. The first shard contains metadata and zero tensors; the remaining descriptors occupy 38,304, 35,232 and 15,328 bytes.

`UD-IQ3_XXS` is the publisher's dynamic quant name. Actual tensor types are q6_K (295), f32 (638), q8_0 (350), iq3_s (41), iq2_s (82), iq4_xs (3), q3_K (1), q2_K (2). Substitution with uniformly IQ3_XXS or Q3_K tensors would change the model.

| Stage | Target weights GiB | Routed subset GiB | Target state at 1M GiB | All target weights + state GiB | Optional draft weights GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A [0,22) | 50.368978 | 46.705078 | 8.821075 | 59.190054 | 0 |
| B [22,27) | 12.695821 | 12.041016 | 1.766724 | 14.462544 | 0 |
| C [27,45) | 46.419694 | 43.576172 | 8.804352 | 55.224046 | 2.607678 |

These are tensor payload sizes and persistent numerical-state arrays, not a Windows admission result. The all-resident column is a lower bound for an all-weights-resident placement; desktops may later keep cold experts on local SSD. RAM and VRAM are separate physical budgets. The apparent laptop headroom does not prove residency, working-set locking, WDDM availability or allocation success.

Per DSA layer, FP16 latent MLA requires `1048576 * 512 * 2 = 1 GiB`; index rows require `1048576 * 128 * 3 * 2 = 0.75 GiB`. Eleven trunk DSA layers therefore need 19.25 GiB total, split 8.75 / 1.75 / 8.75 GiB. BF16 rows have the same byte size. One optional DSA NextN context would add another 1.75 GiB at 1M before metadata/workspace; it is not enabled. The published draft block has no private token embedding: terminal MTP would additionally require a local copy of the input table, or an explicitly validated FP32 token-embedding input from A. The manifest reports the conditional mirror byte count instead of hiding this dependency in the draft total.

Source `llama_hparams::n_embd_r/n_embd_s` gives each KDA layer `3*(4-1)*64*128 = 73728` convolution values and `64*128*128 = 1048576` recurrent values. `llama_model::create_memory` uses FP32 for both. One sequence with no speculative rollback needs 4,489,216 bytes/layer. The 17 / 4 / 13 KDA split contributes the remaining state in the table. Speculative recurrent snapshots multiply those arrays by `1 + n_rs_seq`; they are not free and are not yet planned into runtime admission.

Still to measure/account: host and device mirror copies, Windows/display reserves, actual GPU budgets, cache-cell metadata and pool maps, allocator alignment, CUDA graphs/scratch, prefill high water, expert cache leases, pinned staging, network queues, vision, prefix checkpoints and bounded snapshot buffers. The planner reports `admission=NOT_VALIDATED`. It cannot start inference or claim a 1M allocation succeeded.

Regenerate without weights:

```sh
python3 scripts/glm-fetch-metadata.py /path/to/new-headers
build/bin/llama-glm-plan --validate-schema --output memory.json /path/to/new-headers/*.header
```

Use headers in lexical shard order. Explicit `--split 21,6,18` or neighboring complete splits can be compared; defaults never shrink the context. B must preload every assigned expert, establish resident RAM/VRAM storage, warm/touch required state and fail closed on unestablished residency or later pressure. Mmap alone is insufficient. No runtime provider implementing this guarantee exists yet.

Measured metadata-only comparisons at the same 1M context are saved in `docs/models/glm53-splits.json`. Columns below are all target weights plus numerical state, in GiB, before all runtime/OS reserves:

| Split A/B/C | A | B | C |
| --- | ---: | ---: | ---: |
| 22/5/18 | 59.190 | 14.463 | 55.224 |
| 21/5/19 | 56.647 | 14.463 | 57.767 |
| 22/4/19 | 59.190 | 11.919 | 57.767 |
| 21/6/18 | 56.647 | 17.006 | 55.224 |
| 23/5/17 | 61.733 | 16.209 | 50.934 |
| 20/6/19 | 54.104 | 17.006 | 57.767 |

The 23/5/17 middle range owns two DSA layers, unlike 22/5/18. No final split or Windows residency result is inferred from this comparison. The candidate remains 22/5/18 pending allocator, residency and workload measurements.

With `--validate-schema`, the merged descriptor table is checked by the native GLM loader, including the declared NextN block, using zero-sized weight buffers. The exact pinned headers pass with 1,412 tensors and 120,358,051,192 logical payload bytes. This checks required names/shapes, recognized optional tensors and complete count without reading weights. It does not enable MTP or verify file payloads. Without the option, native_tensor_schema remains NOT_VALIDATED. Neither result grants runtime admission.

The activation bridge now has an explicit local input-buffer cap, at most 128 MiB, and processes one incoming frame at a time (at most 4 MiB). Send slices the existing FP32 model output and allocates only one encoded frame. This bounds transport buffers but is not total runtime admission: B may retain both a whole input and whole output, while graph/state/model buffers and OS socket accounting also coexist. Reserve those lifetimes before production READY. The 128 MiB loopback test proves CPU transport/buffer behavior only.

The new memory expert store limits raw payload bytes and entry count, checks a complete caller-provided owned catalog at seal, and returns immutable leases. It has no disk callback and never evicts. Map/vector/shared-owner overhead remains separate, and ordinary heap pages can still be paged. This is a preload/ownership primitive, not laptop admission. Native expert files read at most 64 MiB per call in 1 MiB OS operations; they allocate only a bounded path/event context, with bytes supplied by the caller. A bounded desktop raw-expert cache now exists; model-copy integration and measured total admission are still required.

Desktop cache reservations include pending loads and retained payloads, even after eviction when a weak-lock race retains the data. Bytes/entry counts release only from the payload deleter, so cache replacement cannot reuse a still-charged allocation. Cold reads are capped at eight concurrent calls and 32 deduplicated waiters; a full leased cache refuses without fetching or skipping experts. Payload counters exclude allocator/map/condition/event overhead and OS file-cache pages. Native ReadFile/pread counts are not proof of physical SSD transactions. GPU hot-cache/copy integration, pinned staging and total physical-reserve measurements remain open.

The GGML expert-bank primitive bounds three raw-quant projection banks by the backend allocator size query, including alignment/padding, before data allocation. It preserves each tensor type and refuses a distinct expert set larger than its slot capacity. Source slabs are leased through synchronous tensor_set copies; the compute lease retains the bank and synchronizes the backend before replacement. The desktop stage loader now uses these banks in actual model callbacks under an aggregate per-context cap, while full routed tensors remain unallocated. The model report separates logical routed bytes, ordinary allocated buffers and raw host payload reservations. This is not an allocator for the whole model or a WDDM/residency guarantee. Retained backend LRU slots now avoid hot-group copies under the same bank cap. Multi-wave prefill, physical GPU qualification and measured admission remain absent; eight slots explicitly require microbatch 1 for the target's eight selected experts. See STAGE_STREAMING.md.

An optional native host-payload lock policy now uses dedicated OS pages, separate page-rounded budgets and final-lease lifetime accounting. Windows verifies working-set Valid/Locked bits; Linux uses real mlock plus resident-page queries. It protects payload bytes only and is not wired into the model API yet. Catalog/allocator overhead, dense/shared weights, attention/KDA/scratch, CUDA staging and GPU residency still require complete laptop admission. Do not present the existing buffered B fixture as locked. See RESIDENT_PAYLOADS.md.

The new rank-1 model API now connects that policy to the complete owned routed expert catalog. It preflights raw bytes, page-rounded bytes and entry count before locks/reads, then retains no inference file reader. The additional CPU fixture locks 4,718,592 bytes below the cloud quota and proves model parity after source deletion. This does not protect ordinary weights/state/workspace or admit physical GPU resources; full residency remains false. Default context capacity is unchanged. See PRELOADED_EXPERTS.md.

A generic locked GGML host-buffer primitive now passes local operator/scheduler and concurrent reservation/lifetime checks under a shared page cap. It can back buffers larger than the expert read cap while preserving that cap for file slabs. It is not integrated into model/cache/graph/output/bank allocation yet. CPU backend work buffers, metadata, network activation buffers and CUDA staging also need separate coverage; routing tensor buffers alone will not establish complete laptop residency. See HOST_BUFFERS.md.
