# GLM stage protocol contract

Status: experimental transport components and a constrained three-process CPU model fixture are implemented. Production workers/coordinator, production identity/config negotiation, command dispatch and session transactions remain absent. The version-1 components do not imply an installable three-PC service.

The future protocol must negotiate protocol major/minor, exact model/quant/source identities, complete three-rank plan, range ownership, tensor schema, FP32 activation transport, FP16/BF16 sparse storage, FP32 KDA state, context capacity, resource limits and authenticated peer identity. READY requires three real stages with validated local tensors, allocations and residency policies. A missing B is a startup failure.

Use explicit integer encodings and bounded frames rather than serialized native structs/pointers. Envelope fields include kind, session, epoch, transaction, chunk, rank, absolute position, rows, dtype, dimensions, declared byte length, state generation and error code. Validate products, lengths, quotas and identity before allocation. FP32 mHC values must preserve the `[hidden,stream,row]` layout. Integer positions remain lossless. Authenticate before accepting expensive allocation requests, do not log secrets/prompts, and do not expose remote command or file-path execution.

Required lifecycle families: HELLO/CAPABILITIES/PLAN/READY; BEGIN/RESET/PREFIX_PROBE/PREFIX_ACCEPT; PREFILL_CHUNK/STAGE_INPUT/STAGE_OUTPUT/STATE_ACK; DRAFT/VERIFY/COMMIT/ABORT; CANCEL/CANCEL_ACK/ERROR/HEARTBEAT; SNAPSHOT_PREPARE/SAVE/SEAL/RESTORE/RESTORE_ACK. Unknown or unsupported capabilities must refuse clearly rather than fake success.

Normal execution traverses all A/B/C target stages. Terminal sampling is not a global commit until all state owners agree. Record committed history, input cursor, valid cache positions, speculative rows and response cursor separately. Detect duplicate and stale transactions; do not re-execute a committed payload after a retry. Uncertain commit invalidates the session and requires reset/replay. Cancellation keeps healthy weights loaded but cannot reuse uncertain mutable state.

Control/cancel must have a bounded path independent of large activation transfers. Reserve queue memory and apply backpressure. Fragment a 128 MiB prefill boundary into independently bounded pieces with deadlines, without allowing unlimited peer-driven buffers.

Snapshots require all three stage records, compatible model/split/precision/schema, bounded streaming and an atomic complete manifest. Partial snapshots must never be advertised as restorable. Prefix identity includes exact token/position history, image identity, template, modes and state generations. Equal image token counts do not imply equal prefixes. MTP stays off until accepted-prefix commit and rejected-future rollback are verified through all stages.

Reference reviewed: StrataRPC exp29 `src/core/layer_stage_rpc.cpp` and `docs/LAYER_STAGE_RPC.md` at `9bd85e5b0ad0f71cabc578fa5c38f5c438bbd5a7`. Its two-rank Qwen protocol, 2 GiB frame ceiling and native model pack are not a drop-in GLM three-rank implementation. Reuse invariants, not incompatible state layouts.

## Implemented frame components

Build with `-DLLAMA_GLM_RPC=ON`. `tools/glm-rpc` contains the C++17 crypto/framing library and `test-glm-wire`. Windows uses the system CNG BCrypt SHA256/HMAC and random providers, linking bcrypt.lib; Linux CPU fixtures use OpenSSL libcrypto (1.1.1 or newer). Provider failure is an error. There is no weak/random fallback, embedded application secret or external crypto DLL requirement on Windows.

All integer fields are explicitly little-endian. The fixed header is 160 bytes:

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | `LLGLMRPC` magic |
| 8 / 10 / 12 / 14 | 2 each | major 1, minor 0, header size 160, message kind |
| 16 / 20 | 4 each | sender / receiver rank |
| 24 / 32 / 40 / 48 | 8 each | epoch / transaction / chunk / absolute first position |
| 56 | 16 | session ID |
| 72 | 8 | state generation |
| 80 / 84 | 4 each | total rows / hidden width |
| 88 / 90 | 2 each | stream count / element dtype |
| 92 / 96 / 100 | 4 each | fragment index / fragment count / this payload length |
| 104 / 112 | 8 each | total logical payload bytes / directional frame sequence |
| 120 | 8 | reserved, must be zero |
| 128 | 32 | HMAC-SHA256 tag |

An activation frame requires FP32, all four streams, the negotiated hidden width, at most 2,048 rows, and positions within the negotiated context (default 1,048,576). The exact total is rows * hidden * streams * 4, capped at 128 MiB. Payload bits are opaque to this codec; the future model bridge must provide little-endian IEEE binary32 `[hidden,stream,row]` bytes without rounding. Fragment payloads are at most 4 MiB (negotiable downward to 64 KiB); count, index and last-fragment length must match that fixed partition. A receiver inspects the fixed header before allocating a payload, then authenticates the complete frame before decoding, execution or acknowledgement. Unknown versions, reserved fields, dtypes, ranks, impossible lengths and overflowing coordinates refuse. The whole-chunk bridge below owns one locally budgeted float input, capped at 128 MiB, plus a bounded frame.

Control messages have no tensor dimensions, exactly one fragment and at most 16 KiB. They use a distinct control-channel key and codec path; activation messages cannot use it. Native TCP tests now exercise separate control/data sockets, completing an authenticated cancel acknowledgement while data writes are backpressured and the peer reads no activation payload. This does not implement model cancellation or distributed state acknowledgement. Recognized kind numbers are a framing vocabulary; application handlers/capabilities are not implemented or advertised.

Directional keys are HMAC-SHA256 of the 32-byte configured PSK over the zero-terminated ASCII domain `LlamaGLM RPC directional key v1`, server nonce (32), client nonce (32), canonical agreed identity digest (32), sender rank (LE32), receiver rank (LE32) and channel (one byte, data=1/control=2). Frame tags authenticate the zero-terminated domain `LlamaGLM RPC frame v1`, header bytes 0..127 and the exact payload. Empty/all-zero secret/binding inputs are refused. Keys are cleared when directional codec objects are destroyed.

Frame sequences start at 1, are strictly consecutive and cannot wrap. Invalid input permanently invalidates the receiving codec; reconnecting must use new nonces and keys. This prevents frame replay within a connection; transaction/session/epoch deduplication across retries and connections remains separate work. HMAC provides authentication and integrity, not transport encryption. No public listener is enabled by these components.

## Implemented mutual proof components

`server_auth` and `client_auth` implement three fixed 160-byte messages. Their local configuration fixes both ranks, channel and canonical identity digest; the socket driver must derive that digest from the complete agreed model/quant/split/precision/schema/limits contract. The canonical local identity encoder below is implemented; remote configuration and production negotiation remain absent. Do not construct directional codecs or admit work until mutual proof completes and the server's final proof has been sent successfully.

Authentication envelopes use magic `LLGLMHS1` at 0, LE16 major/minor at 8/10 (1/0), phase/channel bytes at 12/13, two reserved zero bytes, sender/receiver LE32 at 16/20, server nonce at 24 (32 bytes), client nonce at 56 (32), identity digest at 88 (32), eight reserved zero bytes at 120, and a 32-byte tag at 128. The tag is HMAC-SHA256 with the configured PSK over the zero-terminated domain `LlamaGLM RPC handshake v1` and bytes 0..127. Phase and ranks are authenticated, preventing reflection between client/server proofs or the frame protocol.

The server generates a fresh system nonce for phase 1, signs the challenge and leaves the client nonce zero. The client authenticates it and checks the complete envelope/identity before generating its own fresh system nonce and sending phase 2. The server verifies both nonces/identity and sends phase 3, binding the fresh client nonce to the server's proof. The client exposes an agreed binding only after checking that final proof. A replayed challenge cannot complete against a fresh client nonce. The server exposes its binding after validating phase 2; its network owner must still finish sending phase 3 before use. Both classes refuse wrong state, role, identity, channel, key, reserved fields or proof. Failure is permanent, and handshake PSK copies are cleared on completion/failure/destruction. The application's configured secret remains needed for new connections and directional key derivation.

These components enforce nonce creation and proof state locally. Production model/config and secret provisioning, active connection quotas and actual three-rank startup remain integration work. Deterministic keys/nonces in tests are public fixtures, never runtime defaults.

## Implemented TCP components

`tcp_socket` uses nonblocking Winsock/WSAPoll on Windows and POSIX poll on Linux. Endpoints require explicit numeric unicast IPv4 addresses; wildcard/multicast/broadcast-class addresses, DNS names and invalid connect ports refuse. Listener backlog is bounded to 1..8; it is not an active-connection quota. Port 0 is supported for local fixture listeners. Windows uses exclusive address binding, and all created/accepted handles disable inheritance. RAII closes handles, including configuration/accept failures.

Connections request TCP_NODELAY and 1 MiB send/receive buffers. Actual OS accounting may differ and must enter admission budgets. There is no unbounded user-space send queue. Exact reads/writes handle partial transfers, interruption, EOF and nonblocking readiness against an absolute steady-clock deadline. One deadline covers all three authentication messages. Every I/O/protocol failure shuts down the stream and permanently marks it stopped, preventing queued bytes from reviving a partial connection. The connection owner must discard its codecs and reconnect with fresh mutual proof. `interrupt()` may run concurrently without closing/reusing a handle under another thread; join socket users before destruction or move. Listener interruption is checked at intervals of at most one second.

`authenticate_client/server` return bindings only after complete proof exchange; the server finishes sending its final proof before returning. `read_frame` reads the fixed header, validates limits before allocating at most one bounded payload, reads exactly that body and authenticates it before returning. `write_frame` encodes one bounded packet and writes it against the same deadline. They expose no model paths, commands, remote pointers or payload logging.

Eight local TCP suites use actual loopback sockets in one process. They verify mutual proof and bit-preserving two-way transfers (a 4 MiB fragment plus a 64 KiB tail), independent control under data backpressure, deadline poisoning, partial EOF, wrong secrets, oversized lengths refused without a body, explicit interfaces/port conflict and concurrent interruption. They neither execute models over TCP nor qualify three real processes. Active-peer limits, topology/READY, epoch/transaction deduplication, production model dispatch, all-owner cancellation/reset and serving remain required integration work.

## Three-process CPU fixture

`test-glm-pipeline-worker` is deliberately test-only: fixed six-layer synthetic model/2-2-2 split, 256 context, CPU execution, loopback interfaces and a canonical identity derived from its complete checked descriptors, actual bounded header, source revision, split and modes. The harness configures a fresh random 32-byte secret through a private temporary key file; no credential value enters command arguments, logs, source or retained artifacts. The binary is not installed as a production worker and rejects real models before stage loading. It accepts no model paths or commands from a network peer. The original profile loads buffered owned weights at microbatch 32. A second profile streams A/C experts through the native file/cache/bank provider while keeping B buffered, using a separate monolithic microbatch-1 oracle. Worker records verify zero startup expert-source reads, deferred logical weights and bounded actual inference reads. B has no desktop source, but its buffered heap does not establish physical residency. Each rank initializes its own owned-layer context and processes seven known chunks with checked positions, session/epoch/generation and chunk IDs. A embeds tokens; B/C accept only the actual preceding rank's four-stream tensors. Only C computes logits. Outputs are compared with a separate monolithic generator, which never executes inside A/B/C.

The fixture now uses the whole-chunk bridge with 64 KiB frames; its 64-row chunk crosses two authenticated frames before the next model stage runs. It has no production control/cancel or transaction protocol. Missing B cannot be bypassed by pointing A at C: mutual proof refuses the rank mismatch and no inference output is produced. All sockets/processes/temp files are test-owned and cleaned on success or failure. Production identity/config provisioning, secret provisioning, topology/READY, request commits, all-owner reset/replay and safe admission remain separate required implementation.

## Canonical identity component

`stage-identity` accepts locally checked facts, validates bounds and returns at most 32 KiB of canonical bytes plus a system SHA256 digest. It is not a peer deserializer or an admission decision. Complete native schema validation and consistent local file/header facts must precede its use. The test worker now follows that path; it no longer supplies a constant identity digest. Production file/config provisioning remains unimplemented.

All integers use unsigned little-endian encoding. Strings have a LE32 byte count followed by ASCII bytes (no terminator), capped at 255 bytes except the fixed 40-character lowercase Git revisions. The ordered encoding is:

1. Zero-terminated ASCII domain `LlamaGLM execution identity v1`; LE32 identity version 1; LE16 wire major/minor 1/0.
2. Strings: full configured source revision, architecture `glm5-next`, model repository, model revision, publisher quant name.
3. LE32 tensor-schema version, state-layout version, tensor count; LE64 logical payload bytes.
4. LE32 target layers, declared NextN layers, total/selected/shared experts, and all three exclusive layer ends.
5. LE32 hidden width, stream count, row bound, context capacity, fragment byte cap and control byte cap; LE64 logical chunk byte cap.
6. Strings `f32le:hidden,stream,row`, `f16:mla,index`, `f32:kda`, `causal:seq0:append-only`; LE32 zero feature flags (MTP, vision, prefix reuse and snapshots unsupported).
7. LE32 shard count, followed by each shard in index order: LE32 index, basename string, LE64 full file size, LE64 hashed header length, 32 raw SHA256 header bytes.

All three nonempty ranges must cover the complete target; limits obey the wire component's ceilings. Only version-1 tensor/state layouts are accepted. Shards must have consecutive unique indices, distinct basenames even under Windows ASCII case folding, positive bounded headers and plausible file/payload sizes. Absolute/relative paths and zero header fingerprints refuse. Enumeration order does not affect the digest. No peer controls a weight allocation through this component.

Header fingerprints cover tokenizer/template metadata and tensor descriptors, including quant types and shapes, without reading large tensor payloads. Full file size, publisher revision and exact quant name are bound separately. This is compatibility identity, not proof of payload integrity. The source revision is captured at CMake configuration from Git, or explicitly provided by source-archive builds with `LLAMA_GLM_SOURCE_REVISION`; reconfigure after changing source revisions. It is not binary attestation or proof of a clean source tree. A matching digest cannot establish Windows residency, budgets or healthy model state.

Local independent SHA/canonical-vector tests pass. The actual three-process fixture refuses a tokenizer-header change with unchanged geometry and weights before producing model output; native Windows CNG/CPU qualification passes at 1c1dade88. Production READY and session safety remain separate gates.

## Whole-chunk activation bridge

`send_boundary` and `receive_boundary` provide synchronous forward A->B/B->C model-buffer handoff. The caller supplies locally expected session, epoch, transaction, chunk, generation, absolute position, rows and negotiated limits. The helpers derive byte/fragment counts and refuse impossible shapes or topology. Sender uses slices of its existing FP32 output; it does not create a second whole boundary. Receiver authenticates each bounded frame, requires identical complete chunk coordinates and consecutive fragment indices, then fills one float input capped by an explicit local reservation (at most 128 MiB). It returns only after all fragments match. Little-endian IEEE binary32 preserves every bit; non-little-endian hosts use bounded conversion.

A single absolute deadline covers the whole send or receive, including post-copy checks. Any bad coordinate/order, truncation, MAC failure, insufficient local reservation, timeout or allocation failure discards the partial input and permanently interrupts that socket. A new connection and authenticated state policy are required; there is no partial-frame resume or automatic transaction retry. A completed input does not commit model/cache state or reconcile other owners. Production transaction/reset/cancel handling remains absent.

The local reservation covers the input float buffer only. Admission must additionally account for one frame (up to 4 MiB), encoder packet/header, actual socket buffers, retained model output and graph state, and concurrent input/output lifetimes. The component has no user-space send queue. Control uses separate connections, as qualified by the earlier backpressure fixture.

Eight local loopback suites include a full 128 MiB boundary over 32 frames, short tails at the final context positions, complete-input barrier, coordinate/interleaving/repeat refusal, truncated/bad-MAC final fragments, local budget refusal and whole-chunk deadlines. The actual model fixture now exercises two-frame chunks with byte-exact results. All eight bridge suites and the fragmented model fixture pass native Windows CPU/CNG at f11238d49 in all three build variants.


## Quiescent stage invalidation primitive

llama_context_stage_invalidate is an experimental side-header API for an owner whose in-flight work has already stopped. It idempotently and permanently marks only that stage context failed, suppresses its output views and refuses further evaluation before expert work. Clearing memory cannot revive it; destroy/recreate the context against retained owned weights. Null/ordinary contexts return false. The call is not a running-work interrupt or distributed transaction/reset protocol; a future coordinator must stop/join work, invalidate every affected owner and establish a new epoch/replay before admission.

The local independent model oracle now invalidates A/B/C after valid history (including an already failed C), checks idempotent null outputs and failed retries after memory clearing, and requires unchanged bank materialization/copy counters on rejected retries. It then releases old contexts before allocation, recreates every owner with retained weights and compares the first token's two boundaries/logits against saved independent monolithic outputs. Buffered, streamed and source-deleted locked-host variants pass all seven GLM tests. Ordinary oracle outputs remain usable. Exact-source Windows qualification is pending.

Invalidation suppresses future getter access; callers must discard any previously borrowed output pointers and any uncommitted copies. A local flag cannot retract data already delivered to a peer/client. Production output publication must follow the global commit protocol, which remains unimplemented.

## Owner transaction metadata component

The new stage-session owner ledger checks semantic retries, contiguous positions, complete three-owner commit certificates and newer reset epochs. It retains only fixed-size metadata and does not decode peers, authenticate receipts, execute models or cache replies. A local commit is insufficient for client publication; all three commit acknowledgements and coordinated uncertain-state reset remain unimplemented. See SESSION_LEDGER.md.

Transaction control now has checked fixed applied-receipt and complete-decision payloads; see SESSION_LEDGER.md for the exact fields. Authenticate each control frame before decoding. Control peers form A-B/A-C pairs, applied receipts address A, and decisions originate only from A. Activation data remains A-B-C. An applied STATE_ACK is distinct from a future certificate-bound commit acknowledgement. These components do not yet drive real graph transactions or authorize client publication.

A subsequent metadata component adds distinct commit_ack envelope kind 14 and 256-byte kind-3 payloads containing one local committed receipt plus the complete decision fingerprint. decode_committed_ack requires an already authenticated B/C-to-A control envelope. The coordinator barrier requires all three distinct matching applied receipts and then all three commit ACKs before exposing a committed certificate; it does not publish a reply or reset live owners. See SESSION_LEDGER.md.

The fourth restricted model-process fixture connects the ledgers/barrier to real CPU graph calls and separate A-B/A-C control pairs. Distinct COMMIT_COMPLETE envelope kind 15 has a 176-byte kind-4 payload: prefix, full common key and complete-certificate fingerprint. Only A sends it after all commit ACKs; B/C check it against their own committed decision before releasing diagnostic output. Lost B ACK invalidates all actual owners with zero output; fresh-process replay is exact. This is test integration, not production provisioning, reply caching or retained-weight distributed recovery. See TRANSACTION_FIXTURE.md.
