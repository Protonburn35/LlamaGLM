# Three-stage GLM architecture

Status: partial implementation. Ownership/planning code exists; stage execution and network services do not.

The initial plan is A [0,22), B [22,27), C [27,45). A owns token embeddings and future matching GLM5V preprocessing; C owns final mHC collapse, normalization, output head and optional one-block NextN. The planner rejects empty stages, gaps, overlap, missing coverage and unknown global tensors. NextN block 45 is distinct from target layers 0 through 44. Split counts are configurable and must cover the complete target model.

The existing graph in `src/models/glm5-next.cpp` initializes `inpL` as `[4096,4,rows]`, applies attention and FFN mHC transforms inside each layer, and collapses the four streams only after layer 44. The boundary is after `l_out`, before the next layer's `build_hc_pre`. A contiguous FP32 boundary has 65,536 bytes/token, or 128 MiB for 2,048 rows. A protocol must fragment this into bounded frames. Token positions and pool metadata remain lossless integers. Do not transfer the collapsed `[4096,rows]` head representation at an intermediate boundary.

`build_kda_layer` owns recurrence and Q/K/V short-convolution history per recurrent layer. `build_dsa_layer` owns latent MLA and indexer state per DSA layer. The current indexer stores key, gate and pooled key in each cache row. Every audited trunk DSA layer is a full indexer, so there is no cross-stage indexer-selection reuse in the published schedule. Absolute positions, incomplete four-token pools and recurrent rollback snapshots must remain coherent across chunk boundaries.

The next executor change must extend the existing evaluator with explicit input/output boundary tensors and a validated layer range. Weight loading and memory filters must use exactly the same range. Simply shortening the graph loop would still load unowned tensors and allocate unowned caches; that is insufficient. Keep a monolithic reference path and compare intermediate boundaries/logits using independent deterministic CPU fixtures before network integration.

Persistent native Windows processes will form A -> B -> C, with a separate bounded authenticated control path to avoid cancellation starvation. C samples; A emits only globally committed output. One active request is sufficient. Startup must require all three matching, capable stages. Future desktop weight providers may stream local SSD experts; B's inference provider must have no SSD fallback and must refuse admission unless all required weights/state have a proven residency policy.

Speculation remains off until full-target A/B/C verification and rollback are tested. The existing monolithic MTP implementation is a source reference, not evidence of distributed MTP. Vision, prefix reuse, cancellation and three-stage snapshots likewise require new integration and validation.
