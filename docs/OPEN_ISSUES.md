# Open engineering work

1. P0: no owned-layer loader/graph/executor yet. Integrate range filters for tensors and all three memory components; add four-stream graph inputs/outputs and independent boundary parity fixtures.
2. P0: no three-rank authenticated TCP transport or transaction state machine. Implement bounded framing, control isolation, epoch/dedup, cancellation and uncertain-state reset before any user-facing candidate.
3. P0: desktop local SSD tier and laptop resident-only weight provider are absent. Prove Windows RAM/VRAM admission, no hidden full-model copies, and no inference-time laptop paging/SSD fallback.
4. P1: memory manifest excludes allocator/cell metadata, Windows/WDDM reserves, scratch/graphs, queues, vision, prefix/snapshot buffers and speculative peaks. Current byte totals are not admission guarantees.
5. P1: parser checks geometry, descriptors, complete shard count and basic layer/global coverage; it is not yet a complete expected-per-layer tensor/shape validator. Add structural inventory fixtures before using the manifest to admit an executor.
6. P1: Windows CPU/CUDA compile and packaging are not yet observed. Validate native tools and dependencies via the exact Actions SHA. CUDA runtime DLL redistribution and full launchers remain distribution work.
7. P1: distributed API, tokenizer/template integration, tools/reasoning streaming, image identity, prefix reuse, manual snapshots and MTP verification/rollback are unimplemented.
8. P1: real quant numerical fidelity, local GPU execution and all physical three-PC tests remain not run. No performance claim is justified.
