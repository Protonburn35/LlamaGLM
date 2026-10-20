# GLM stage protocol contract

Status: design contract; no network codec or listener is implemented yet. The stage-plan schema is version 1 and does not imply a working wire protocol.

The future protocol must negotiate protocol major/minor, exact model/quant/source identities, complete three-rank plan, range ownership, tensor schema, FP32 activation transport, FP16/BF16 sparse storage, FP32 KDA state, context capacity, resource limits and authenticated peer identity. READY requires three real stages with validated local tensors, allocations and residency policies. A missing B is a startup failure.

Use explicit integer encodings and bounded frames rather than serialized native structs/pointers. Envelope fields include kind, session, epoch, transaction, chunk, rank, absolute position, rows, dtype, dimensions, declared byte length, state generation and error code. Validate products, lengths, quotas and identity before allocation. FP32 mHC values must preserve the `[hidden,stream,row]` layout. Integer positions remain lossless. Authenticate before accepting expensive allocation requests, do not log secrets/prompts, and do not expose remote command or file-path execution.

Required lifecycle families: HELLO/CAPABILITIES/PLAN/READY; BEGIN/RESET/PREFIX_PROBE/PREFIX_ACCEPT; PREFILL_CHUNK/STAGE_INPUT/STAGE_OUTPUT/STATE_ACK; DRAFT/VERIFY/COMMIT/ABORT; CANCEL/CANCEL_ACK/ERROR/HEARTBEAT; SNAPSHOT_PREPARE/SAVE/SEAL/RESTORE/RESTORE_ACK. Unknown or unsupported capabilities must refuse clearly rather than fake success.

Normal execution traverses all A/B/C target stages. Terminal sampling is not a global commit until all state owners agree. Record committed history, input cursor, valid cache positions, speculative rows and response cursor separately. Detect duplicate and stale transactions; do not re-execute a committed payload after a retry. Uncertain commit invalidates the session and requires reset/replay. Cancellation keeps healthy weights loaded but cannot reuse uncertain mutable state.

Control/cancel must have a bounded path independent of large activation transfers. Reserve queue memory and apply backpressure. Fragment a 128 MiB prefill boundary into independently bounded pieces with deadlines, without allowing unlimited peer-driven buffers.

Snapshots require all three stage records, compatible model/split/precision/schema, bounded streaming and an atomic complete manifest. Partial snapshots must never be advertised as restorable. Prefix identity includes exact token/position history, image identity, template, modes and state generations. Equal image token counts do not imply equal prefixes. MTP stays off until accepted-prefix commit and rejected-future rollback are verified through all stages.

Reference reviewed: StrataRPC exp29 `src/core/layer_stage_rpc.cpp` and `docs/LAYER_STAGE_RPC.md` at `9bd85e5b0ad0f71cabc578fa5c38f5c438bbd5a7`. Its two-rank Qwen protocol, 2 GiB frame ceiling and native model pack are not a drop-in GLM three-rank implementation. Reuse invariants, not incompatible state layouts.
