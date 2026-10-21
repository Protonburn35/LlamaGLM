# Actual stream and owned-graph telemetry

The TCP component now exposes saturating atomic stream counters through tcp_socket::io_stats(). Counters follow the underlying socket across moves; an empty wrapper returns zero. They count actual positive recv/send results, including partial transfers that later fail. They never count the requested byte length as completed traffic.

| Fields | Meaning |
| --- | --- |
| read_bytes, written_bytes | Actual TCP stream bytes, including mutual-proof messages and frame headers/payloads |
| recv_calls, send_calls | Actual native calls, including EOF/would-block/interrupted/error results |
| read_waits, write_waits | Calls that wait/check readiness before a native transfer; one may poll internally several times |
| read_wait_ns, write_wait_ns | Steady-clock time inside those readiness checks, including immediate polls, OS scheduling and peer delay |
| read_failures, write_failures | Failed valid read/write API calls; later calls on a poisoned stream count separately |
| timeout_calls | Failed I/O calls classified as deadline/native timeouts |

Bytes exclude TCP/IP/Ethernet headers, retransmission costs and kernel buffers. Readiness time is not isolated network latency and can include waiting for another stage's compute. Counters do not cover connect/accept duration, protocol/MAC validation failures after successful I/O, or socket memory occupancy. Invalid API arguments rejected before I/O are not native I/O failures. Values saturate at UINT64_MAX; they do not wrap.

Each field is atomic so one reader, one writer, an interrupter and telemetry snapshots may coexist within the existing socket lifetime contract. A concurrent snapshot is approximate across fields; do not infer cross-field consistency until users have quiesced/joined. Move/destruction still requires stopped/joined users. Telemetry does not synchronize model state or change failure semantics.

The model-process fixture reports per-completed-epoch data-in/data-out/control snapshots. Control counters accumulate discarded healthy connections before fresh authentication, preserving actual retry handshake traffic. A recovery epoch starts fresh counters; failed prior-epoch lifetime totals remain separate future work. The worker also measures wall time around its actual owned llama_process calls and total epoch wall time from context construction through completed graph/control/output work. Model preload is outside that epoch wall time. Compute time includes expert-source/copy work inside decode; attention/MoE/device breakdowns remain unimplemented.

The Python harness compares every successful process against an independently calculated byte ledger. For the small hidden-128/four-stream 131-token model, each normal forward connection writes 269,728 bytes: 268,288 activation bytes, eight 160-byte frame headers and the client's 160-byte proof. The server sends 320 proof bytes in reverse. The healthy committed retry adds a 2,048-byte boundary and one 160-byte header, giving 271,936 forward bytes. B's aggregate data directions also include the other connection's reverse proof.

The committed fixture's two A-B control handshakes and eight BEGIN/decision rounds produce A read/write totals 7,040/9,792 bytes. Each BEGIN adds 144 payload bytes plus the 160-byte header. A-C adds eight 432-byte terminal-result frames, giving A read/write totals 10,496/9,792. Reverse endpoint counters must match. All three earlier profiles and every retry/recovery/cursor profile check the complete ledger with zero successful-epoch I/O failures.

Nine TCP suites qualify successful authenticated bytes, partial EOF counts, deadlines/poisoned calls, concurrent snapshots during a real 4 MiB transfer and counter ownership across a socket move. Use `bash scripts/glm-net-sanitizer.sh address` or `thread` for standalone Linux ASan/UBSan or TSan checks; both pass without reports. The full local thirteen-CTest/seven-GLM gate passes. Exact-source native telemetry qualification is pending.

These are synthetic CPU observations, including CUDA-linked builds that execute CPU fixtures. They are not a target-PC benchmark or a throughput guarantee. GPU timings, real model quantization, physical link behavior, TTFT/client output rates, attention/MoE subphases, queue occupancy, complete memory admission, clocks/thermals and full 1M execution remain unqualified. Socket/telemetry metadata and kernel buffers remain explicit residency/accounting gaps.

Native telemetry qualification passes at 0ab6c9746 before BEGIN was introduced (7,360 control write bytes per A peer). The later local BEGIN extension raises that exact ledger to 9,792 and has its own pending native qualification.


Exact local sampling setup/ACK adds two 272-byte framed writes and reads per A-peer control pair, one on each authenticated connection. The latest seeded 33-row profile writes 22,176 per A-peer, reads 15,584 from B and 24,224 from C (320-byte outcomes), while 33-row greedy reads 23,360 from C (272-byte outcomes). Forward data stays 105,440 with the repeated 17-row prompt. Independent ledger checks cover these new messages; see SAMPLED_RESULTS.md.
