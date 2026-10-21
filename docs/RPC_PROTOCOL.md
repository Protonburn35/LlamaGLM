# GLM stage protocol contract

Status: experimental authenticated frames and fixed-size mutual proof components are implemented and pass CPU component tests. There is no listener, socket handshake driver, fragment assembler, command dispatcher or transaction coordinator yet. The stage-plan schema and wire format both start at version 1; neither implies a working network service.

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

An activation frame requires FP32, all four streams, the negotiated hidden width, at most 2,048 rows, and positions within the negotiated context (default 1,048,576). The exact total is rows * hidden * streams * 4, capped at 128 MiB. Payload bits are opaque to this codec; the future model bridge must provide little-endian IEEE binary32 `[hidden,stream,row]` bytes without rounding. Fragment payloads are at most 4 MiB (negotiable downward to 64 KiB); count, index and last-fragment length must match that fixed partition. A receiver inspects the fixed header before allocating a payload, then authenticates the complete frame before decoding, execution or acknowledgement. Unknown versions, reserved fields, dtypes, ranks, impossible lengths and overflowing coordinates refuse. No full 128 MiB reassembly allocation is present yet.

Control messages have no tensor dimensions, exactly one fragment and at most 16 KiB. They use a distinct control-channel key and codec path; activation messages cannot use it. Separate sockets, bounded queues and deadlines still need implementation to make cancellation independent in practice. Recognized kind numbers are a framing vocabulary; handlers/capabilities are not implemented or advertised by an application.

Directional keys are HMAC-SHA256 of the 32-byte configured PSK over the zero-terminated ASCII domain `LlamaGLM RPC directional key v1`, server nonce (32), client nonce (32), canonical agreed identity digest (32), sender rank (LE32), receiver rank (LE32) and channel (one byte, data=1/control=2). Frame tags authenticate the zero-terminated domain `LlamaGLM RPC frame v1`, header bytes 0..127 and the exact payload. Empty/all-zero secret/binding inputs are refused. Keys are cleared when directional codec objects are destroyed.

Frame sequences start at 1, are strictly consecutive and cannot wrap. Invalid input permanently invalidates the receiving codec; reconnecting must use new nonces and keys. This prevents frame replay within a connection; transaction/session/epoch deduplication across retries and connections remains separate work. HMAC provides authentication and integrity, not transport encryption. No public listener is enabled by these components.

## Implemented mutual proof components

`server_auth` and `client_auth` implement three fixed 160-byte messages. Their local configuration fixes both ranks, channel and canonical identity digest; the socket driver must derive that digest from the complete agreed model/quant/split/precision/schema/limits contract. There is no remote configuration or identity negotiation handler yet. Do not construct directional codecs or admit work until mutual proof completes and the server's final proof has been sent successfully.

Authentication envelopes use magic `LLGLMHS1` at 0, LE16 major/minor at 8/10 (1/0), phase/channel bytes at 12/13, two reserved zero bytes, sender/receiver LE32 at 16/20, server nonce at 24 (32 bytes), client nonce at 56 (32), identity digest at 88 (32), eight reserved zero bytes at 120, and a 32-byte tag at 128. The tag is HMAC-SHA256 with the configured PSK over the zero-terminated domain `LlamaGLM RPC handshake v1` and bytes 0..127. Phase and ranks are authenticated, preventing reflection between client/server proofs or the frame protocol.

The server generates a fresh system nonce for phase 1, signs the challenge and leaves the client nonce zero. The client authenticates it and checks the complete envelope/identity before generating its own fresh system nonce and sending phase 2. The server verifies both nonces/identity and sends phase 3, binding the fresh client nonce to the server's proof. The client exposes an agreed binding only after checking that final proof. A replayed challenge cannot complete against a fresh client nonce. The server exposes its binding after validating phase 2; its network owner must still finish sending phase 3 before use. Both classes refuse wrong state, role, identity, channel, key, reserved fields or proof. Failure is permanent, and handshake PSK copies are cleared on completion/failure/destruction. The application's configured secret remains needed for new connections and directional key derivation.

These components enforce nonce creation and proof state locally. TCP exact reads/writes, deadlines, connection limits, explicit allowed interfaces, canonical identity construction, secret-file/environment provisioning and actual three-rank startup remain integration work. Deterministic keys/nonces in tests are public fixtures, never runtime defaults.
