# Autoregressive CPU fixture

The fifth test-only pipeline profile feeds actual committed C selections back to A. It uses the existing synthetic F32 model, 2/1/3 owned-layer split, 16 experts/all eight selected, four FP32 residual streams, context 256 and microbatch 1. It does not provide a production server or change target model/sampling defaults.

The separate monolithic oracle starts with a 17-token prompt. It selects the first maximum from each final FP32 logit row and uses that token for the next decode. Sixteen single-token steps follow the prompt, for 33 processed rows and 17 terminal selections. The last selection is checked but not consumed. This independent argmax path has the same lowest-token-ID tie rule as C's inherited native greedy sampler.

The harness gives A only the original 17 prompt tokens. B/C plans contain no token field. All owners receive the fixed prompt/decode coordinates, not the expected generated sequence. After each all-owner commit and authenticated C result, A appends that returned token to its runtime token history, hashes the actual next input for BEGIN, and evaluates its own layers. The expected generated sequence remains only in the independent oracle/harness. Every consumed token, both boundaries, all logits, complete certificate history and each compact token/top-eight result must match exactly.

The first completed prompt is retried after fresh control authentication. Its cached 17-row boundary and terminal result must repeat without another graph call or cursor advance. Each owner performs exactly 17 graph calls and 33 bank materializations per owned MoE layer. C sends 18 fixed 272-byte outcome payloads including the retry. Each forward data stream carries 105,440 actual bytes including proof/frame overhead; reverse data proof is 320 bytes. Per A-peer control writes are 21,632 bytes; A reads 15,040 from B and 22,816 from C. Actual native counters are checked against that independent ledger.

A second attempt drops B's commit ACK at chunk index 5, after a globally completed 21-row prefix. All three invalidate their old contexts and unwind their old replies/connections. The same processes retain each owned model, accept a fresh controller session/newer epoch, recreate contexts and regenerate from the prompt using new actual C results. The existing response journal verifies the 21 old output rows without appending them, then appends 12 new rows. Final boundaries/logits contain exactly 33 rows. Each model loads once; each owner creates two contexts. B's source stays deleted and its model-native read count does not increase during inference or recovery.

This qualifies deterministic greedy feedback and synthetic replay. Stochastic sampler/RNG transactions, complete request/configuration identity, production reset/admission/HTTP, physical GPU/three-PC execution, the real mixed-quant model and full 1M context remain unimplemented or NOT_RUN. Full B working-set residency stays false.


Native Windows qualification passes at c6dd1be50, run 39587356972, on CPU/SM89/SM120a hosted builds. Thirteen CTest targets, seven GLM tests and real-header schema pass; downloaded CPU evidence confirms both autoregressive attempts. A later extension routes C greedy selection through cloned native sampler transactions and expands the local gate to fourteen CTest targets. See TRANSACTIONAL_SAMPLING.md.


A later explicit sampled profile now exercises native temperature/distribution feedback and RNG replay through the same real model graphs. All transactional control connections add exact 112-byte configuration setup/ACK before BEGIN and after reconnect, so the latest control byte totals supersede the earlier figures above. See SAMPLED_RESULTS.md.
