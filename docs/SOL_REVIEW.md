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

Remaining review gates: Windows/CUDA qualification of the owned-layer graph/loader, strict laptop residency, expert lease lifetimes, native authenticated transport and transaction recovery, full API/prefix/snapshot state, and release dependencies/launchers. These require implementation and new focused tests before a broad candidate review is possible.

Owned-loader extension reviewed: the complete three-range plan is copied per model; only GLM5Next is admitted; unknown file tensors remain visible to strict loader accounting; unowned entries are removed before ordinary payload reads, with no whole-file mmap/prefetch. Generic optional tensor passes see null unowned layer pointers. Stable public parameter structs are unchanged; the C interface lives in an experimental side header. Byte comparisons and non-GLM regression exercise both modified and untouched paths.

Local evaluator extension reviewed: all three memory filters share weight ownership; B/C input and A/B output widths preserve every FP32 mHC stream; C alone collapses streams. Headless outputs bypass pooling/logit/sampling work and retain every row. Output-ID and memory inputs are built only when used, including pure-KDA/DSA splits. Ordinary monolithic context creation still rejects partial models. The stage factory/defaults and decode guards bound context, enforce FP16 sparse storage, causal seq-0 append-only positions and unsupported-feature refusal. An RAII guard permanently invalidates a context after uncertain preparation/compute, and a CPU abort fixture proves a memory clear cannot silently reuse it. Independent monolithic boundary captures and logits qualify both split layouts locally. Remaining gates include three-process transactions, exact Windows/CUDA numerical results, the real mixed quant, 1M allocations and physical residency; local fixture parity does not establish those.

Frame component review: no native struct serialization, unchecked peer-sized allocation or cryptographic primitive implementation. Dimensions/counts/positions are bounded before products and payload reads; full frame bytes are authenticated with native providers. Context identity, channel, nonces and direction separate link keys. Sequence advancement follows successful authentication; any receiver failure is permanent. Tests include independently encoded vectors and mutations, rather than only encoder-to-decoder roundtrips. No listener or application accepts this codec yet. Fresh-nonce mutual handshake, authenticated identity/limit negotiation, fragment ordering/assembly, deadlines/queues and session/transaction semantics remain mandatory integration work.
