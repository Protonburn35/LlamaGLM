# GLM stage protocol contract

Status: experimental frames, mutual proof and nonblocking TCP drivers are implemented and pass local component/loopback tests. There is no worker application, canonical identity negotiation, fragment execution/assembly, command dispatcher or transaction coordinator yet. The stage-plan schema and wire format both start at version 1; neither implies a working three-stage service.

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

Control messages have no tensor dimensions, exactly one fragment and at most 16 KiB. They use a distinct control-channel key and codec path; activation messages cannot use it. Native TCP tests now exercise separate control/data sockets, completing an authenticated cancel acknowledgement while data writes are backpressured and the peer reads no activation payload. This does not implement model cancellation or distributed state acknowledgement. Recognized kind numbers are a framing vocabulary; application handlers/capabilities are not implemented or advertised.

Directional keys are HMAC-SHA256 of the 32-byte configured PSK over the zero-terminated ASCII domain `LlamaGLM RPC directional key v1`, server nonce (32), client nonce (32), canonical agreed identity digest (32), sender rank (LE32), receiver rank (LE32) and channel (one byte, data=1/control=2). Frame tags authenticate the zero-terminated domain `LlamaGLM RPC frame v1`, header bytes 0..127 and the exact payload. Empty/all-zero secret/binding inputs are refused. Keys are cleared when directional codec objects are destroyed.

Frame sequences start at 1, are strictly consecutive and cannot wrap. Invalid input permanently invalidates the receiving codec; reconnecting must use new nonces and keys. This prevents frame replay within a connection; transaction/session/epoch deduplication across retries and connections remains separate work. HMAC provides authentication and integrity, not transport encryption. No public listener is enabled by these components.

## Implemented mutual proof components

`server_auth` and `client_auth` implement three fixed 160-byte messages. Their local configuration fixes both ranks, channel and canonical identity digest; the socket driver must derive that digest from the complete agreed model/quant/split/precision/schema/limits contract. There is no remote configuration or identity negotiation handler yet. Do not construct directional codecs or admit work until mutual proof completes and the server's final proof has been sent successfully.

Authentication envelopes use magic `LLGLMHS1` at 0, LE16 major/minor at 8/10 (1/0), phase/channel bytes at 12/13, two reserved zero bytes, sender/receiver LE32 at 16/20, server nonce at 24 (32 bytes), client nonce at 56 (32), identity digest at 88 (32), eight reserved zero bytes at 120, and a 32-byte tag at 128. The tag is HMAC-SHA256 with the configured PSK over the zero-terminated domain `LlamaGLM RPC handshake v1` and bytes 0..127. Phase and ranks are authenticated, preventing reflection between client/server proofs or the frame protocol.

The server generates a fresh system nonce for phase 1, signs the challenge and leaves the client nonce zero. The client authenticates it and checks the complete envelope/identity before generating its own fresh system nonce and sending phase 2. The server verifies both nonces/identity and sends phase 3, binding the fresh client nonce to the server's proof. The client exposes an agreed binding only after checking that final proof. A replayed challenge cannot complete against a fresh client nonce. The server exposes its binding after validating phase 2; its network owner must still finish sending phase 3 before use. Both classes refuse wrong state, role, identity, channel, key, reserved fields or proof. Failure is permanent, and handshake PSK copies are cleared on completion/failure/destruction. The application's configured secret remains needed for new connections and directional key derivation.

These components enforce nonce creation and proof state locally. Canonical identity construction, secret-file/environment provisioning, active connection quotas and actual three-rank startup remain integration work. Deterministic keys/nonces in tests are public fixtures, never runtime defaults.

## Implemented TCP components

`tcp_socket` uses nonblocking Winsock/WSAPoll on Windows and POSIX poll on Linux. Endpoints require explicit numeric unicast IPv4 addresses; wildcard/multicast/broadcast-class addresses, DNS names and invalid connect ports refuse. Listener backlog is bounded to 1..8; it is not an active-connection quota. Port 0 is supported for local fixture listeners. Windows uses exclusive address binding, and all created/accepted handles disable inheritance. RAII closes handles, including configuration/accept failures.

Connections request TCP_NODELAY and 1 MiB send/receive buffers. Actual OS accounting may differ and must enter admission budgets. There is no unbounded user-space send queue. Exact reads/writes handle partial transfers, interruption, EOF and nonblocking readiness against an absolute steady-clock deadline. One deadline covers all three authentication messages. Every I/O/protocol failure shuts down the stream and permanently marks it stopped, preventing queued bytes from reviving a partial connection. The connection owner must discard its codecs and reconnect with fresh mutual proof. `interrupt()` may run concurrently without closing/reusing a handle under another thread; join socket users before destruction or move. Listener interruption is checked at intervals of at most one second.

`authenticate_client/server` return bindings only after complete proof exchange; the server finishes sending its final proof before returning. `read_frame` reads the fixed header, validates limits before allocating at most one bounded payload, reads exactly that body and authenticates it before returning. `write_frame` encodes one bounded packet and writes it against the same deadline. They expose no model paths, commands, remote pointers or payload logging.

Eight local TCP suites use actual loopback sockets in one process. They verify mutual proof and bit-preserving two-way transfers (a 4 MiB fragment plus a 64 KiB tail), independent control under data backpressure, deadline poisoning, partial EOF, wrong secrets, oversized lengths refused without a body, explicit interfaces/port conflict and concurrent interruption. They neither execute models over TCP nor qualify three real processes. Active-peer limits, topology/READY, epoch/transaction deduplication, fragment ordering/execution, all-owner cancellation/reset and serving remain required integration work.
