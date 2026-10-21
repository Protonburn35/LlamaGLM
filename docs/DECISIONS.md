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
