# Authenticated request agreement

A now introduces each transaction's actual request fingerprint through an authenticated BEGIN command to B and C. Workers retain local authority over identity/session/epoch/generation, next transaction/chunk IDs, absolute position, row count and bounds. They do not need the original token plan to process an intermediate activation.

The existing BEGIN envelope kind is 4. Its fixed 144-byte payload uses domain GLMTXN, format version 1, payload kind 6 and zero reserved fields, followed by the existing 128-byte complete step key. Integers remain unsigned little-endian. The key includes the canonical execution identity, session/epoch/generation, transaction/chunk/position/rows and nonzero original-request fingerprint.

Authenticate the entire control frame before decoding. The decoder requires A as sender and B or C as receiver; exact fixed length, domain/version/kind, reserved fields and key bounds apply. Every state coordinate must match the independently supplied local begin_coordinates. The returned request digest comes from A. The envelope must also match the complete decoded key, empty control tensor shape, unfragmented lengths and nonzero verified sequence. No request-controlled allocation occurs here.

A permitted new request can have a different digest; workers must not require prior knowledge of its token bytes. Duplicate handling remains semantic: the complete returned key must match owner-ledger/cached history before reusing output. Changed request hashes for the same committed transaction invalidate the owner/epoch rather than re-executing. BEGIN is request agreement, not applied state, resource admission or global commit. The current fixture sends no separate BEGIN admission ACK and cannot authorize production READY.

The fourth model-process profile sends BEGIN before A computes/sends activations. B/C check it against their local planned coordinates, receive/authenticate the actual four-stream boundary, then hash that input before owner prepare. Applied receipts and complete commit certificates bind the agreed request hash through the entire A-B-C chain. A healthy reconnected committed retry repeats BEGIN and requires exact key equality before forwarding retained bytes.

B/C now receive a coordinates-only test plan with the token field removed. Only A reads the original fixture tokens. All ordinary, lost-ACK, process-restart, retained-weight and committed-response cursor profiles still match the independent monolithic boundaries/logits and compact terminal outcomes. This is a falsifiable removal of shared-token knowledge, not a production configuration/HTTP interface. Row scheduling, geometry and test-controller provisioning remain constrained fixture inputs.

The fixture hashes actual little-endian int32 token bytes; other fields are separately bound by the complete key and immutable source/model identity. Production must additionally bind complete sampling/generation settings and other request semantics before enabling them. Stateful sampling/RNG, dynamic HTTP requests, client history/image/tool data and production authenticated reset remain absent.

Fourteen session suites include independent Python golden BEGIN bytes, all truncated lengths, domain/reserved/state/role/phase refusal, new request-hash establishment and changed-duplicate rejection. The session suite passes ASan/UBSan, and the full local thirteen-CTest/seven-GLM gate passes. Independent byte ledgers include eight 304-byte BEGIN frames per control peer (144 payload plus 160 header). Native exact-source BEGIN qualification is pending.


Native qualification passes at fadd5e3a2, Windows run 39586278010, on CPU/SM89/SM120a hosted builds. Thirteen CTest targets, fourteen session suites, seven GLM tests and real-header schema pass. Runtime evidence is CPU only. The later autoregressive fixture hashes actual C-selected next inputs; see AUTOREGRESSIVE_FIXTURE.md.
