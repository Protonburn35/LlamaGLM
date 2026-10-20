# Implementer review notes

This is a self-review of the first planning/CI component, not an independent review or a final production-pipeline signoff. The distributed runtime is still unimplemented.

Reviewed paths: GGUF callback reader, typed metadata access, plan/range arithmetic, tensor routing, state-size arithmetic, expert slab bounds, command-line/output behavior, metadata-only fetch and executable Windows workflows. Applied security, design, testing and general coding checks from the repository's code-review guidance where relevant to this fork.

Findings addressed:

- The GGUF total block count includes NextN; use 45 target layers plus one explicit draft block.
- The publisher uses INT32 KV-head arrays, while synthetic fixtures can use UINT32. Check the array element type and count before selecting the matching cast.
- Bound header bytes and fixed-header tensor/KV counts before the upstream reader allocates metadata containers. Refuse incomplete or duplicate shards/tensors and unknown global names. Do not claim tensor data or admission was verified.
- Check byte arithmetic before addition/multiplication. Expert file spans include all 288 slabs, enforce matrix orientation/shape and reject invalid expert IDs or overflowing offsets. Never change routing to fit a cache.
- Require dense/routed/shared FFN projections at all target/NextN layers. A full attention/KDA schema validator remains outstanding and is recorded in OPEN_ISSUES.
- Account for C's optional MTP dependency on A's token embedding table. MTP stays disabled.
- Windows internal architecture fixtures require static llama libraries. Keep the fixture; fix the build configuration rather than deleting checks.
- Windows workflow permissions are contents-read, third-party Actions are pinned, CUDA archives retain official SHA256 verification, CCCL has a specific commit, no weights or credentials enter artifacts, and hosted CPU tests are not called GPU validation.

Remaining review gates: the complete owned-layer graph/loader, CUDA numerical qualification, strict laptop residency, expert lease lifetimes, native authenticated transport and transaction recovery, full API/prefix/snapshot state, and release dependencies/launchers. These require implementation and new focused tests before a broad candidate review is possible.
