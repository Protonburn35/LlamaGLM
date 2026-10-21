# Three-process transaction fixture

The fourth pipeline profile connects actual owned-layer CPU graphs to the owner ledgers, authenticated transaction codecs and coordinator commit barrier. It remains a test worker restricted to the small synthetic F32 model, loopback interfaces, fixed fixture geometry and per-run secret files. It is not a production worker, admission path or serving API.

The independent monolithic oracle uses the existing small 2/1/3 split, context 256, hidden 128, four residual streams, 16 routed experts/all eight selected, microbatch 1 and two CPU threads. A/C stream expert slabs; B preloads its complete 48-slab catalog and policy-2 host tensors/state/workspaces. The harness removes B's private same-basename model copy before A connects. Full working-set admission remains false: metadata/control/network allocations and GPU resources are not protected by this fixture.

Activation data follows A-B-C. Separate authenticated duplex control connections join A-B and A-C. Each control handshake has fresh mutual nonces; forward/reverse keys bind the same handshake to their distinct sender/receiver ranks, channel and identity. Each direction has its own frame sequence. No default credential is embedded. A new process group receives a fresh 16-byte session, credential and explicit generation. All workers use the same canonical model/header/split/source identity.

```mermaid
sequenceDiagram
    participant A as A / coordinator
    participant B as B
    participant C as C
    A->>B: four-stream boundary
    B->>C: four-stream boundary
    B->>A: authenticated applied receipt
    C->>A: authenticated applied receipt
    Note over A: include actual local A receipt; validate full chain
    A->>B: complete commit decision
    A->>C: complete commit decision
    Note over A: commit A locally
    B->>A: certificate-bound committed ACK
    C->>A: certificate-bound committed ACK
    Note over A: require all three distinct committed ACKs
    A->>B: authenticated global completion
    A->>C: authenticated global completion
    Note over A,C: release local diagnostic output only after completion
```

Each chunk's key binds the complete session/epoch/generation, transaction/chunk/absolute position, rows and SHA256 of its actual little-endian token bytes. B/C independently know those fixture requests; production BEGIN/request provisioning is still absent. Owners prepare before graph execution and mark applied only after successful synchronous decode and hashing actual unfragmented FP32 output bytes. B/C hash their actual received activation bytes. The complete certificate checks every owner and both boundary digest links. C's output fingerprint covers its actual local logits.

A sends decisions only after all applied receipts agree, commits locally, then waits for matching B/C commit ACKs. COMMIT_COMPLETE has envelope kind 15 and a fixed 176-byte kind-4 payload: the usual 16-byte prefix and 128-byte common key followed by the complete canonical decision's 32-byte SHA256. Only A sends it, after commit_barrier::committed() succeeds. A worker decodes it only after its own local commit, matching the complete key and decision fingerprint. Applied ACKs, decisions, committed ACKs and completion notices cannot substitute for each other. Independent Python golden bytes, exact lengths, reserved fields, altered fingerprints and unauthorized roles are tested.

Local diagnostic output files remain empty until a chunk's global completion. The worker retains its current context output without starting another decode while control is pending. Seven committed retries return metadata without another graph call; actual execution/retry counters and identical certificate histories are compared across all three processes. Every boundary and logit byte is checked against the independent oracle over seven uneven chunks/131 tokens. A subsequent reply-cache extension now captures bounded immutable copies before applied acknowledgement, uses native-locked reply pages on B and reads committed retries/diagnostic files from those copies. This still does not supply production reconnect/retry dispatch or a client response protocol; see REPLY_CACHE.md.

The failure case makes B commit the first step locally and close without sending its committed ACK. A cannot cross the all-three barrier. A/B/C interrupt their connections, permanently invalidate their quiescent actual contexts and expose no residual/logit getters. All three exit with zero diagnostic output and no completed transaction report. The harness then starts new processes with fresh credentials/session and generation 2, replays from position zero and requires exact oracle output. This reloads owned weights; distributed retained-weight context recreation, replay of prior client-visible history, response/RNG cursors and running-work cancellation remain unimplemented.

The fourth profile passes native Windows at 6928167bb. The later reply-cache extension has its own pending native qualification.

The three earlier pipeline profiles and their missing-middle/changed-tokenizer refusal cases remain. Linux passes the full twelve-CTest/seven-GLM gate with the fourth profile, lost-ACK refusal and process-restart replay. The twelve session component suites also pass standalone ASan/UBSan. Exact-source native Windows qualification of this extension is pending. Real model quantization, GPU, physical PCs and full 1M execution remain NOT_RUN.

The next retry extension deliberately replaces both healthy control connections after the first known global completion. Fresh mutual proof, directional keys and frame counters are required. A resends its cached boundary; B/C hash the duplicate actual input, require the committed disposition, and return/forward the original immutable reply. The identical certificate/commit ACK/completion exchange repeats without another graph call, cursor advance or output append. The harness requires one actual network retry per owner and unchanged 131-token bank materializations. Uncertain/lost-ACK transport never uses this path. Native qualification of this reconnect extension is pending.

## Retained-weight recovery profile

A later profile keeps all three processes and owned models alive after the lost first-step B commit ACK. The per-epoch scope invalidates its actual context and replies, interrupts connections and unwinds every context/codec/lease before announcing RESET_READY. B checks tracked host buffer counts/reservations return to the pre-context model-only baseline and its private model source is still absent. All three diagnostic outputs must be zero before the test controller proceeds.

The controller sends a bounded JSON line through each child's stdin pipe, not a network reset endpoint. The exact six-field version-1 plan supplies a fresh 16-byte session, strictly newer epoch/generation, next data port and A's B/C control ports. It contains no credential. The same per-run PSK authenticates entirely fresh data/control handshakes/nonces; old connections/keys/counters and old epoch replies are discarded. The controller first provisions C, then B, then A only after every owner reports readiness.

Each worker recreates a context against its retained model, with no model reload or B source restoration. Evidence requires one model load and two actual context creations per same PID, identical owned ranges/bytes, unchanged B startup expert I/O, exact seven-commit/131-token boundaries/logits and the healthy cached network retry. Context input positions restart at zero. This local profile passes the full thirteen-CTest/seven-GLM gate. Exact-source native qualification is pending.

This exercises quiescent context recovery after a first-step uncertain commit with no previously published client history. Production authenticated reset control, reconstruction of prior committed user/assistant history, response/RNG cursors, compute interruption and complete resource admission remain separate requirements. The stdin controller is strictly fixture orchestration.

## Committed response position across replay

The later cursor extension injects B's lost commit ACK after an already globally completed prefix. A fixed seven-entry per-owner fingerprint journal advances only after global completion and survives context destruction. The response row count remains separate from the new epoch's input/state position. Old uncommitted replies are revoked as before.

Before provisioning recovery, the parent checks that output files contain exactly the independent oracle's already committed prefix. Children never read those files to reconstruct inference: the fixed original request tokens and committed output hashes remain in memory, and the output stream only appends. Recreated graphs replay from position zero and verify every prior output fingerprint in order without appending those rows again. New globally completed rows advance response position normally.

Cases with 1 and 56 committed rows require respectively 130 and 75 newly appended rows, exact final 131-row byte contents, one model load/two contexts, B source absence and all previous cache/control checks. The 56-row case spans five uneven chunks and fails during the next 64-row step, exercising KDA/DSA history beyond a single replay row. The local full thirteen-CTest/seven-GLM gate passes; exact-source native cursor qualification is pending. This is a bounded synthetic recovery journal, not a durable conversation store, saved-prefix compute reuse, snapshot format or sampler/RNG journal.

The compact terminal extension now invokes the existing native greedy sampler on C's actual last-row logits after global completion, sends only a 272-byte token/top-eight FP32 outcome to A and reuses it for the actual healthy retry. Independent monolithic full logits verify all seven outcomes/candidate bits in every normal/recovery/cursor profile; B receives no terminal results. Fixed input replay remains; this is not autoregressive generation or stochastic/RNG support. See TERMINAL_RESULTS.md.

Completed epochs now expose actual data/control stream counters (including healthy reconnect proofs) and measured owned CPU graph/epoch wall time. The harness independently checks every byte and successful I/O failure count in normal and recovery profiles. Failed prior epochs and physical/GPU performance remain separate. See RPC_METRICS.md.
