# Local deployment configuration

`llama-glm-config --check deployment.json` validates the local configuration syntax and common three-PC intent. It reads only that bounded UTF-8 JSON file. A successful result has `configuration: VALID` and `admission: NOT_VALIDATED`, with tensor checks `NOT_RUN`, zero model payload bytes read, credentials unread and no network operations. It is a configuration tool, not a worker launcher or resource preflight.

Start from [config/deployment.example.json](config/deployment.example.json). Copy it locally, set the selected rank, local paths and the three actual interface addresses. All peers use the same common topology, capacities and layer split. Each can use different local directories and a different credential-file path. Relative paths resolve against the configuration file's directory, not the shell working directory. Model and credential files are not opened by this command, so a valid report does not establish their existence, contents, permissions or physical locality. Paths are omitted from its report and errors. Windows uses wide command-line arguments and UTF-8 JSON path conversion.

The schema requires version 1, local rank 0/1/2, all three node records, a local credential-file reference and all four original publisher shard filenames in order. Repository `unsloth/GLM-5.3-Flash-GGUF`, revision `a38483c8cd5df544f53d70fb281afe97369d5ab6` and dynamic quant label `UD-IQ3_XXS` must match exactly. No inline credential field or arbitrary alternative model is accepted. The future loader must still validate actual headers/schema/file consistency and the published mixed tensor types; a matching JSON label is not payload verification.

Defaults are the full 1,048,576 context, layer ends 22/27/45, row ceiling 2,048 and frame ceiling 4 MiB. Optional explicit `context`, `row_limit`, `frame_bytes` and `layer_ends` values obey existing protocol bounds and must retain three nonempty contiguous owners covering all 45 target layers. Explicit smaller capacities are reported and change the common intent; no default is silently reduced. The row ceiling is a protocol bound, not a backend microbatch or performance claim.

Addresses must be distinct canonical numeric non-loopback unicast IPv4 interfaces. Wildcards, DNS names, multicast, leading-zero octets and duplicate addresses refuse. A alone has an API port and no incoming stage data/control port; B/C have separate nonzero data/control ports and no API port. Port reuse across different PCs is allowed. No bind/connect is attempted. These syntax checks cannot establish separate physical machines, connectivity, peer identity or ready model state.

Unknown fields, duplicate JSON keys, nesting beyond 16, files over 64 KiB, booleans/fractions for integers, invalid split/port ownership, control bytes, URL paths and UNC/network-style paths refuse. The initial schema also excludes extended Windows path prefixes. Error messages exclude supplied values and parser excerpts, including malformed JSON containing a private literal. Config/model paths are local trusted input; native source opening, fixed-volume checks, symlink/file-race protections and credential ACL/32-byte/nonzero validation remain worker-startup work.

The report's `intent_sha256` is a common configuration fingerprint. It excludes local rank, model directories and credential path/content, and does not attest a binary or model. All three local ranks and reordered node input produce the same digest. Changes to topology, capacities or split change it. Version-1 canonical bytes are:

1. NUL-terminated ASCII `LlamaGLM deployment intent v1`, LE32 version 1.
2. Repository, revision and quant as LE32 byte lengths plus ASCII bytes.
3. LE32 context, row ceiling, frame bytes and the three layer ends.
4. For each rank 0/1/2: LE32 rank, length-prefixed IPv4 string, LE32 data/control/API ports.

The parser produces typed local facts and this digest; callers must keep them immutable after parsing. Future authenticated startup must bind common intent together with actual canonical model/header/source identity and resource policy before READY. The digest itself grants no capability or admission.

The new fifteenth CTest target checks all three owners/defaults, independent Python canonical hashing, reordered nodes, relocated Unicode paths, explicit capacities/split, six intent-changing fields and 39 malformed/privacy refusals. No model or credential file is created for these tests. The full fifteen-CTest/seven-GLM gate and standalone parser/CLI ASan/UBSan pass locally. Exact-source native Windows qualification, including wide Unicode argument handling, is pending.

Production resource admission, protected configuration/control ownership, credential loading, model source inspection, peer provisioning, workers/reset and distributed API remain open. See OPEN_ISSUES.md and RESIDENCY_STATUS.md. Real mixed-quant/GPU/physical three-PC/1M execution remains NOT_RUN.


Native Windows syntax/Unicode qualification passes at 11963acff, run 39594391976, in CPU/SM89/SM120a hosted builds with fifteen CTest/seven GLM/native schema gates. A later extension adds separate bounded model-header inspection/reference auditing; check mode retains its original zero-model-access contract. See MODEL_HEADER_FACTS.md.


An additional explicit --check-key operation now loads the configured raw 32-byte nonzero key under a private native-file policy and verifies dedicated locked pages, with no model/network work. Ordinary check/header modes remain credential-unread and all modes keep admission false. No production key or permission is generated/changed by the tool. See CREDENTIAL_PROVISIONING.md.
