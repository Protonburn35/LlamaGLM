# Checked model header facts

The deployment tool now has three distinct operations:

| Operation | Reads | Establishes |
| --- | --- | --- |
| `--check deployment.json` | Bounded local JSON only | Valid syntax/common intent; no model or credential checks |
| `--inspect deployment.json` | Native local full-file sizes and exact bounded header prefixes | All four local sizes, pinned header fingerprints and native complete tensor schema |
| `--audit-headers deployment.json header1 header2 header3 header4` | Explicit metadata-reference files only | Pinned reference bytes/schema and expected model identity; local full files remain unchecked |

All operations retain `admission: NOT_VALIDATED`. Inspection/audit reports zero tensor payload bytes read, zero schema weight allocation, credentials unread, no network operations and `tensor_payloads_verified: false`. Full local GGUF size checks do not verify weight contents. The cloud/CI positive test uses only reference headers; successful inspection of the real complete local GGUF set remains NOT_RUN.

`stage-model-facts` reuses the native immutable positional file reader and existing complete GLM schema validator. It opens only regular sources; Windows requires a resolved fixed local volume and denies concurrent write/delete handles. It checks source length before allocation/read, then reads exactly the pinned prefix and verifies SHA256 before native GGUF parsing. Header-only input cannot be mistaken for a full-file size check. There is no buffered-file read-ahead into tensor data, mmap or network fetch in this probe.

| Shard | Published full bytes | Header bytes read | Header SHA256 |
| --- | ---: | ---: | --- |
| 1 | 9,429,984 | 9,429,956 | `e47e6ba043a5d406cdf695219ff9728666dc3a0392eb7190171ca13fd671d63d` |
| 2 | 49,261,511,584 | 38,304 | `ad6191c93269faeee16be8d2d1746df56f43e7f6515c26d646368a358f0ad7d4` |
| 3 | 49,075,832,480 | 35,232 | `eed639822ed7b1e16efe6a4ab6d68728b5b3648c3270fe379218fb984097747e` |
| 4 | 22,020,797,792 | 15,328 | `269b5ce802b9e424ad10d9781c709eace77e3310e5e1ba63939a3c074bfb20e3` |

The first shard has no tensors and has 28 unused bytes beyond the hashed metadata prefix. Total actual metadata reads are 9,518,820 bytes. Prefix buffers are parsed in memory with no tensor allocation. Every declared tensor must be unique and fit its published source bounds. The complete 1,412-descriptor table, including the separate NextN block, passes the native loader schema and exact 120,358,051,192-byte logical payload inventory. Header hashes cover actual mixed types and tokenizer/template metadata; no uniform quant replacement occurs. The metadata fetch script now also checks these exact fingerprints before saving its bounded HTTPS responses.

Checked identity binds the complete source/model/header/split/precision/bounds record through the existing canonical identity component. An independent Python encoder reconstructs its exact bytes/hash. Deployment intent remains a separate common topology/capacity fingerprint. Local paths and credential material are omitted from reports and neither digest proves payload integrity, a clean binary or resource residency.

Returned facts retain opaque shared native source handles until the last facts owner is released. A future worker must keep actual source identity coherent through loading, then release B's file sources after complete preload. Holding a probe's handles is not a complete load transaction or protection against every later reopen/path change. Reference handles are explicitly marked by `full_file_sizes_checked: false`; their published sizes are compatibility reference data. They must never grant production startup permission.

The existing fifteen-CTest/seven-GLM gate passes with additional missing/incorrect full-file refusals. Separate real-metadata regression tests qualify default/reordered-rank/explicit-capacity identities plus reordered, changed-byte, truncated and missing reference refusals. Mutations reject before native GGUF parsing. No sparse or fabricated complete weight files are created. Standalone component/CLI ASan/UBSan also passes these seven cases while linking the qualified core; it does not instrument the entire core. A standalone sanitizer link initially omitted ggml-base and was corrected before successful execution.

Windows Actions now run this audit after fetching the pinned headers, compare independent identities/refusals and retain `model-header-facts.json` beside the planner manifest. Exact-source native qualification of this extension is pending. Model payload verification, credential loading, resource/GPU probes, complete protected startup ownership, workers/READY/reset/API and real mixed-quant/GPU/physical three-PC/1M execution remain open or NOT_RUN.


Native Windows qualification passes at 056c17d2f, run 39597405259, on CPU/SM89/SM120a hosted builds. Fifteen CTest/seven GLM/native schema and independent header-facts/refusal gates pass; downloaded model-header-facts.json confirms 1,412 descriptors/9,518,820 metadata bytes/reference-only/full-file-false/admission false. The later credential provider is separate.
