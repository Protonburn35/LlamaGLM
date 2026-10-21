# Compact committed terminal results

The transaction codec and fourth synthetic process profile now carry a compact C-to-A greedy outcome after global commit. C uses the existing native llama_sampler_init_greedy/llama_sampler_apply implementation on its actual last-row logits. It sends the selected token and eight diagnostic raw FP32 candidates. Full vocabulary logits stay local; the complete local logit file is only the independent test artifact.

Version 1 supports stateless greedy mode only. Unsupported sampling modes refuse. This does not implement stochastic sampling, penalties, grammar, probability normalization, EOS/client streaming policy, RNG commit/recovery or autoregressive feedback. The fixture still processes its fixed known input tokens. Its locally expected vocabulary/candidate count and mode are explicit; production must bind the complete sampling configuration/seed to request/session agreement before adding other modes. No ordinary llama sampler behavior changes.

TERMINAL_RESULT is appended envelope kind 16. Its fixed 272-byte kind-5 payload uses the existing explicit little-endian encoding:

| Region | Bytes | Fields |
| --- | --- | --- |
| Prefix | 16 | GLMTXN domain, version 1, kind 5, reserved zero |
| Common key | 128 | Full identity/session/epoch/generation/transaction/chunk/position/rows/request fingerprint |
| Decision | 32 | SHA256 of the complete canonical three-owner commit certificate |
| Result fields | 32 | LE32 rank 2, vocabulary, selected token, selected raw FP32 bits, mode 1, candidate count, last-row index, reserved zero |
| Candidates | 64 | Eight fixed LE32 token/FP32-bit pairs; unused entries must be all zero |

Vocabulary is positive and bounded to 1,048,576 token IDs; this limit is independent of context length. Candidate count is 1..8 and cannot exceed vocabulary. Only the last row of the exact pending step is accepted. Tokens are in range and unique. Scores are finite raw logits, sorted descending with increasing token ID for ties; the selected greedy token/score exactly matches the first candidate. The float bits remain IEEE binary32, not decimal values or normalized probabilities. Unknown modes, nonfinite values, wrong fixed length, reserved fields and noncanonical unused entries refuse.

Authenticate the complete frame before decoding. The decoder requires C-to-A roles, exact control envelope/key, the locally committed certificate fingerprint and explicitly expected vocabulary/candidate count. Structural validation is not a proof that an argmax was computed; A trusts its authenticated configured C owner. Independent model/oracle tests qualify that implementation. Canonical execution identity remains compatibility identity, not binary/runtime attestation.

C computes the first outcome only after receiving A's global completion, with its own actual committed reply. A reads it only after its all-three commit barrier succeeds. The previous outcome is retained for the healthy committed network retry, producing byte-identical result without another sampler or graph call. A and C record the same seven outcomes; B receives none. Seven normal steps plus one actual network retry carry 2,176 result payload bytes per direction endpoint, plus framing/TCP overhead. Each result frame adds the existing 160-byte header; full vocabulary arrays are never returned over LAN by this path.

The independent monolithic full-logit oracle supplies the selected token and all eight token/FP32-score pairs at every chunk's final row. All results are compared exactly, including replay/retry profiles and 1/56-row committed-prefix recovery. Thirteen session component suites include an independent Python golden hash, every short length, wrong role/phase/key/certificate/configuration, reserved/unused fields, duplicate/unsorted candidates and NaN/infinity refusal. The standalone session suite passes ASan/UBSan. The full local thirteen-CTest/seven-GLM gate passes. Exact-source native terminal-result qualification is pending. No production worker/API, resource admission or GPU/full-model qualification is implied.


The legacy format above remains greedy-only. A separate sampled format now binds exact agreed configuration and committed native RNG ordinal and permits selected tokens outside raw top-eight diagnostics. Real seeded model feedback/recovery is qualified locally; see SAMPLED_RESULTS.md.
