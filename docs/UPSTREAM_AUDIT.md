# Pinned upstream audit

Base: `1242be66f65d5d8a3d996ba56d6de106ced98cf6` (2026-10-20). Verified origin `Protonburn35/LlamaGLM`, parent `ggml-org/llama.cpp`, public, default master. Existing master is left unchanged. Git read and Actions run listing work; the integration cannot read the Actions administration-policy endpoint (403). Workflow execution must be established by an actual push/run.

Inspected symbols: `llama_model_glm5_next::{load_arch_hparams,load_arch_tensors,build_arch_graph}`, `graph::{build_kda_layer,build_dsa_layer,build_hc_pre,build_hc_post}`, `graph_mtp`, `llama_model::create_memory`, `llama_memory_hybrid_idx`, `llama_memory_recurrent`, `llama_hparams::{n_embd_r,n_embd_s}`, `llama_moe_cache::impl` and `moe_cache_slru`. The monolithic trunk loops through all target layers and collapses streams at the head; neither local GPU tensor splitting nor existing backend RPC provides the required three-process owned-layer product.

Verified each merged commit below is an ancestor of the pinned base with `git merge-base --is-ancestor`. Retrieved actual file patches and available issue/review discussions. Relevant examined behavior is summarized below; open changes have not been imported or exhaustively qualified.

| PR | Observed state | Merge SHA (merged) or head SHA (unmerged) | Decision/source observation |
| --- | --- | --- | --- |
| [#27773](https://github.com/ggml-org/llama.cpp/pull/27773) | merged/in base | `649dcb1036fd47d01a83b56c5e35913e65c277ba` | Graph, GGUF conversion, KDA/DSA/mHC and GLM5V foundation; inherited. |
| [#29928](https://github.com/ggml-org/llama.cpp/pull/29928) | merged/in base | `b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea` | One NextN DSA block; separate target/draft graph and shared-sequence rollback guard; inherited. |
| [#29958](https://github.com/ggml-org/llama.cpp/pull/29958) | merged/in base | `c173a53bdfca1047c710018dc934a6d67a8b010f` | Stable k-pool graph allocation and pooled-row writes; inherited. |
| [#30042](https://github.com/ggml-org/llama.cpp/pull/30042) | merged/in base | `3109914090564b4c5280f30896369d55b86bbdbf` | Sparse attention replaces gather path; inherited. |
| [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) | merged/in base | `d6cf9acb25e6c657d6c6a4645ab7b69fce119995` | GPU cache of host experts, small-batch copy/remap path; inherited. |
| [#30112](https://github.com/ggml-org/llama.cpp/pull/30112) | merged/in base | `c811cb8f0ac91b8ac72a32f970bdd45037f20da7` | Per-device cache groups, still local multi-GPU; inherited. |
| [#30637](https://github.com/ggml-org/llama.cpp/pull/30637) | merged/in base | `33d6135cbe20e90e67d9db1c874fbb4a9dc64b19` | Probation/protected SLRU lists and leases; inherited. |
| [#30848](https://github.com/ggml-org/llama.cpp/pull/30848) | open / unmerged | `a11fac528e1c1dfa75214cacc197f905852dcf2a` | Small-cache threshold and LRU below 1.8x one-token expert set; evaluate before laptop caching. |
| [#30594](https://github.com/ggml-org/llama.cpp/pull/30594) | open / unmerged | `44c74f9ded04b477b9dfa4373f275f92052aa08c` | Profile-prewarmed GPU slots/CPU miss path; requires explicit activation-precision audit, not imported. |
| [#25294](https://github.com/ggml-org/llama.cpp/pull/25294) | open / unmerged | `1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5` | TENSOR_STREAMED and expert-slab I/O/cache waves; Windows shared-handle/direct-I/O semantics need repair before reuse. |
| [#28544](https://github.com/ggml-org/llama.cpp/pull/28544) | open / unmerged | `879f4d67a107c799ec571ee4e42398b971226d73` | Eval-callback expert-ID trace; useful later for actual routing measurements. |
| [#30205](https://github.com/ggml-org/llama.cpp/pull/30205) | open / unmerged | `8f7f1ae0541dc7cedf505b2adb9b8f9bc514481c` | Per-device budgets; one budget per distributed process may avoid this API extension. |
| [#26542](https://github.com/ggml-org/llama.cpp/pull/26542) | open / unmerged | `66f40bc010f4a95b3cfc17903d65467389b32b6e` | Positional reads, device alignment, private Windows handles; synchronous ReadFile, not IOCP. |
| [#26659](https://github.com/ggml-org/llama.cpp/pull/26659) | open / unmerged | `7f77eab4d2569b72b71d790cf439a25fdeb1768d` | Optional mapped-weight registration with permissive fallback; does not establish strict B residency. |
| [#29963](https://github.com/ggml-org/llama.cpp/pull/29963) | open / unmerged | `3ca7eed13274e406822fa1ba3e69a883560566bc` | Local host-expert scheduling/copy lifetime; not network stage RPC. |
| [#24675](https://github.com/ggml-org/llama.cpp/pull/24675) | closed / unmerged | `b628acfef4d3641351ba1be2fc3487201cb59cda` | Closed async backend RPC queue experiment, overlaps another implementation; not a protocol foundation. |
| [#30499](https://github.com/ggml-org/llama.cpp/pull/30499) | open / unmerged | `a68f5ebd6183faf701a930b5bac7af3187cd92e7` | Contiguous permuted queries for tensor split; distinct from contiguous process layer ownership. |
| [#30640](https://github.com/ggml-org/llama.cpp/pull/30640) | open / unmerged | `8a86212329f2d07f76fb3ba3799436e19b4d2205` | Fused k-pool mask op/backend additions; optional optimization after baseline parity. |
| [#30593](https://github.com/ggml-org/llama.cpp/pull/30593) | closed / unmerged | `1daeb59d1dbcd0a4b2f35e63086da54d0a7ebeee` | Closed unmerged; active-expert clipping excluded by fidelity contract. |
| [#30763](https://github.com/ggml-org/llama.cpp/pull/30763) | closed / unmerged | `89a88a34050f6dd4b0288b7c5a004fbef09f0515` | Closed unmerged; cache-biased expert selection excluded. |

The published quant metadata at `unsloth/GLM-5.3-Flash-GGUF@a38483c8cd5df544f53d70fb281afe97369d5ab6` contains 46 blocks including one NextN block, 45 target layers, INT32 KV-head schedule, all 1,412 tensor descriptors and eight mixed tensor encodings. Official config revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` confirms geometry, 288/8 experts, four streams, 1M positions, 34 KDA/11 DSA and GLM5V. Matching `mmproj-F16.gguf` (1,128,047,296 bytes) and `mmproj-BF16.gguf` (1,164,010,176 bytes) are published; no projector payload was downloaded or executed.

StrataRPC exp29 reference: `9bd85e5b0ad0f71cabc578fa5c38f5c438bbd5a7`, `docs/LAYER_STAGE_RPC.md` and `src/core/layer_stage_rpc.cpp`. Reviewed bounded little-endian framing, handshake/spec identity, persistent sessions, worker sampling and paired snapshots. Its two-stage Qwen pack/NextN state and synchronous operation path cannot simply be renamed into GLM three-stage state. No source was copied. Strata's MIT license was inspected; preserve its notices if code is later adopted.

DeepSeekRPC main resolved to `25de683e5ef3a9299267b8e61bafb1d755941c84` for reference only. No code dependency, modifications, cherry-picks or architectural assumptions were taken from it. No other repository is a write target.

License/provenance: retained the fork's existing MIT license and third-party source notices. New planner code uses the existing GGUF reader, ggml quant-size calculations and vendored nlohmann JSON. No new runtime library or copied Strata/DeepSeek source. Windows compiler components are pinned NVIDIA CUDA 12.8.1 redistributables verified against official SHA256 metadata; CCCL is pinned to `e46b01ff821f3badeea858f0124ce00a7dd7d05a` (v3.4.3). An upstream-referenced 13.4.0 redistrib manifest was unavailable (404), so it was not treated as an installable toolchain.
