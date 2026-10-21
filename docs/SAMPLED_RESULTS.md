# Authenticated sampling agreement and results

The sixth synthetic CPU pipeline profile uses the inherited native temperature/distribution sampler with explicit seed 53 and temperature 0.75. A receives only the 17-token prompt; sixteen actual committed C selections drive subsequent decode. A separate monolithic model uses independent native sampler objects and supplies the expected sequence, all 33 boundary/logit rows and all 17 selected outcomes. Sampling uses all 128 vocabulary entries, including actual selections outside the eight diagnostic candidates. This is an explicit fixture policy, not a restriction on future production sampling support.

Every transactional control connection now agrees its locally configured sampling parameters before BEGIN. A sends SAMPLING_SETUP (envelope 18); each distinct B/C peer checks the exact local configuration and returns SAMPLING_SETUP_ACK (19). A waits for both acknowledgements. Both payloads are 112 bytes: `GLMSRPC\0`, LE16 version 1, LE16 kind 8 or 9, LE32 reserved zero, the 64-byte identity/session/epoch/generation stamp, then the 32-byte canonical sampling configuration. Their control envelopes require transaction/chunk/position zero and the authenticated expected A-owner direction. The body kinds cannot substitute for each other. These acknowledgements establish configuration agreement, not resource admission or READY.

Fresh control authentication after known completion repeats agreement against the same immutable local configuration. It does not reset the sampler, owner ledger, cached outcome or RNG ordinal. All actual fixture request hashes now include a domain, the canonical sampling fingerprint and actual LE int32 input tokens. The full step key additionally binds model identity/state coordinates. This covers the supported fixture policy; production image/tool/request semantics still need their own canonical identity.

The original TERMINAL_RESULT (16, 272 bytes) remains greedy-only. Sampled C-to-A results use a separate SAMPLED_RESULT (17) and fixed 320-byte body:

| Byte offset | Field |
| --- | --- |
| 0 | `GLMSRPC\0`, LE16 version 1, LE16 kind 7, LE32 reserved zero |
| 16 | Common 128-byte complete step key |
| 144 | SHA256 of the complete canonical commit certificate |
| 176 | Canonical sampling configuration fingerprint |
| 208 | LE64 committed sampler draw ordinal |
| 216 | Eight LE32 fields: rank C, vocabulary, selected token, raw FP32 selected logit bits, mode, diagnostic count, final row, reserved zero |
| 248 | Eight token/raw-FP32 diagnostic pairs; unused entries zero |
| 312 | Eight reserved zero bytes |

The decoder requires an already authenticated C-to-A control frame, the complete locally committed certificate, expected local configuration and exact ordinal. Every key/certificate/configuration field must agree. Candidate tokens are finite, unique, in range and sorted by raw score descending/token ID ascending. The selected token can lie outside the diagnostic set in distribution mode; its raw score must be consistent with the visible ordering. Greedy still requires the first diagnostic. These checks do not attest an untrusted C binary or prove an unseen distribution. Full-vocabulary logits remain local.

C draws on a pending clone before applied acknowledgement and promotes only after checked global completion. Healthy retries return the same native selection/certificate/ordinal without another draw or graph call. The sampled profile repeats the completed prompt after fresh authentication and checks 18 result transfers of 320 bytes, including that retry. A later lost B ACK invalidates all old context/reply/sampler owners. The same models create new contexts/samplers under a fresh session/newer epoch; actual graph replay reconstructs the seeded RNG path, verifies 21 prior published rows and appends 12 new rows. Final generated history and all boundary/logit bytes remain exact, with one model load/two contexts per process and no B source restored.

A separate test gives B seed 54 while A/C expect 53. Authenticated setup refuses before BEGIN or model output. Nine sampling suites include independent Python configuration/setup/ACK/result golden bytes, every short result/setup length, reserved/role/state/configuration/ordinal checks, nonfinite/duplicate/order refusal, 2,000 bounded mutations with canonical re-encoding, actual HMAC poisoning and forbidden data-channel use. The full fourteen-CTest/seven-GLM local gate and standalone component ASan/UBSan pass.

Each 33-row forward data stream still carries 105,440 actual bytes including framing/proof/retry. Per-peer A control writes are 22,176 bytes after setup on both connections; A reads 15,584 from B and 24,224 from C in sampled mode. The legacy greedy control result path reads 23,360 from C for the same coordinates. Independent byte ledgers verify all actual socket counters. These are CPU/loopback observations, not target link performance.

Production configuration/reset/API, grammar and the other sampling policies, durable client/RNG journals, complete B residency/GPU admission and physical target execution remain open. Real mixed-quant/GPU/three-PC/1M execution remains NOT_RUN.


Native Windows qualification passes at c0ec31052, run 39590421567, for all fourteen CTest/seven GLM/native schema gates on CPU/SM89/SM120a hosted builds. Downloaded CPU evidence confirms seeded native RNG replay and configuration refusal. The later input-buffer extension is separate; see LOCKED_NETWORK_INPUT.md.
