# Engineering decisions

## 2026-10-20

- Preserve the existing fork/history and work only on `codex/glm53-3pc-rpc`; use project-local nonpersonal Git attribution. Master is not an integration target yet.
- Pin upstream `1242be66f65d5d8a3d996ba56d6de106ced98cf6`, Unsloth revision `a38483c8cd5df544f53d70fb281afe97369d5ab6`, and official config revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.
- Reuse the already-merged GLM5Next/MTP/SLRU implementations. No open prototype PR is blindly merged. No expert top-k clipping, cache-biased routing or lossy wire precision is accepted.
- Inventory the exact quant before selecting an executor/placement. The five-layer laptop split has a 14.46 GiB all-target-weights/state lower bound; actual residency remains an engineering requirement.
- Keep KDA state FP32, matching upstream memory creation. FP16/BF16 sparse storage and FP32 transport are distinct precision decisions.
- Separate metadata planning from inference admission. The native planner explicitly reports NOT_VALIDATED for admission and does not claim missing runtime capabilities work.
- Use normal Windows-2022 hosted Actions for native builds, pinned verified CUDA 12.8.1 components and pinned CCCL. No paid infrastructure or physical GPU is provisioned.
- StrataRPC is a read-only design reference, not a source transplant. Its Qwen/two-stage state is not GLM's. DeepSeekRPC main resolved to `25de683e5ef3a9299267b8e61bafb1d755941c84`; no dependency or source import was taken.

## 2026-10-21

- Expose stage context creation through the experimental side header; keep ordinary model/context behavior and stable parameter structs unchanged.
- Use one shared ownership predicate for weights and all three memory components. Preserve all mHC streams through A/B and make boundary widths explicit in batch/output APIs.
- Start with single-sequence, causal, append-only evaluation. Reject unsupported pooling, MTP/rollback, backend samplers and cross-layer shared index selections. Default to full model context, capped at the target's 1,048,576 positions, without a hidden smaller fallback.
- Permanently invalidate a stage context after preparation/compute failure; recovery recreates contexts against retained weights. Distributed coordination/reset/replay remains required before this can serve requests across processes.
- Keep the monolithic evaluator as the oracle. Use independent intermediate captures and final logits, mixed KDA/DSA ownership and incomplete-pool/chunk boundaries to qualify local execution before transport integration.
- Use system CNG on Windows and libcrypto for Linux CPU authentication fixtures; do not implement cryptographic primitives. Bind HMAC keys to fresh connection nonces, canonical identity, direction and channel. Keep control frames independent and small. HMAC integrity is not encryption, and connection sequence checks do not replace session/transaction deduplication.
- Enforce mutual proof with one-use fixed-size handshake components and system-generated nonces. The client cannot expose a binding before the final server proof; all authentication errors permanently invalidate the handshake. Keep socket/identity construction and distributed admission as explicit remaining integration work.
- Use explicit unicast IPv4 interfaces, nonblocking sockets and absolute I/O deadlines. Shut down and permanently stop a failed stream instead of resuming partial frames. Keep socket handles valid during concurrent interruption and join users before destruction. Prove actual control/data independence over separate loopback sockets before connecting model processes.
- Qualify the compute chain with three actual owned-layer processes and a separate monolithic oracle. Keep fixture workers loopback-only with explicit small-model guards and per-run secret files, with no embedded worker credential. They remain separate from the production application until identity, admission and session transactions exist. Require missing-middle startup refusal and exact aggregate weight coverage.

- Validate complete descriptors through a guarded no-allocation native loader mode, rather than duplicating attention/KDA/NextN shape tables. Require real descriptors, including unknown-tensor count rejection; do not use the synthetic provider path as a schema validator. Keep payload verification and runtime admission false after metadata validation.

- Bind the complete execution contract with system SHA256 over explicit little-endian fields and length-prefixed ASCII identifiers. Hash bounded GGUF headers (including tokenizer/template metadata) and declared file sizes/revisions; do not require full-weight checksum passes. Treat this identity as a compatibility check, not payload integrity, binary attestation or resource admission.

- Receive a complete locally planned activation chunk before mutating model state. Authenticate every bounded fragment, require exact semantic coordinates and increasing fragment indices, cap the single input buffer by a local reservation, and permanently interrupt/discard on any error. Keep whole-chunk transport completion distinct from distributed model commit/reset.

- Start SSD work with read-only explicit-offset handles, no shared seek cursor or mmap, exact bounded byte transfers and source-mutation refusal. Keep the sealed memory expert store free of file handles/callbacks; a selected-expert miss fails. Do not equate a sealed heap catalog with Windows page locking or final laptop residency.

- Keep desktop raw-expert caching restricted to ranks 0/2 and the complete locally checked owned catalog. Deduplicate loads and evict only unleased LRU entries. Track payload reservations through final buffer deletion, including weak-lock races and data retained past cache ownership; do not free budget merely because a map entry was evicted. Fatal load failure stops new cache work. This does not change routing or create a laptop fallback.

- Qualify raw-cache-to-GGML bank materialization with real native quantized expert operators before changing model callbacks. Preserve original expert IDs for scales/biases and remap only GEMM slot IDs. Hold an exclusive compute lease through backend completion and poison any partly failed bank. Refuse excess distinct experts explicitly; do not clip router output or claim unimplemented wave execution.

- Integrate desktop sources through metadata-only routed tensors, owned native file offsets and per-context backend banks. Catch every provider exception inside the GGML callback and synchronize before decode checks its permanent failure status. Bound worst-case selected experts by an explicit microbatch instead of silently altering routing; keep context capacity independent. Require CPU-bank op_offload=false to prevent stale scheduler weight copies. Rank 1 continues to refuse this disk-backed API, and buffered heap remains insufficient for laptop admission.

- Preserve the original buffered three-process oracle. Qualify streamed A/C processes with a separate monolithic oracle at the same explicit microbatch 1, rather than relaxing exact boundary checks across different compute schedules. Require startup and final source counters without presenting native calls as physical SSD traffic or buffered B as resident.

- Retain immutable complete expert groups in backend banks. Reserve every selected hot slot before choosing unselected LRU victims, commit mappings only after all projection copies finish, and poison any failed replacement. Report backend reuse/copies separately from raw RAM-cache/native-file events; CPU bank counters are not GPU runtime evidence.
