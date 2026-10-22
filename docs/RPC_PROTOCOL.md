# RPC protocol

LlamaGLM uses the unmodified upstream ggml RPC protocol of llama.cpp b11780 (`ggml/src/ggml-rpc/ggml-rpc.cpp`, protocol v8). All three PCs must run binaries from the same source version. There is no LlamaGLM-specific protocol, state machine or authentication.

## Connections

- Stage A is the only client. It opens one TCP connection per worker (`--rpc 192.168.50.129:50052,192.168.50.143:50053`), does a `HELLO` version check and asks for the device count.
- Each worker serves one client at a time. Its devices come from `-d CUDA0,CPU`; Stage A names them globally in `--rpc` order: B -> RPC0 (GPU), RPC1 (RAM); C -> RPC2 (GPU), RPC3 (RAM). Buffer type names in the log use the worker-local index: `RPC0[192.168.50.129:50052]` = B's GPU, `RPC1[192.168.50.129:50052]` = B's RAM.
- `llama-rpc-test` and `llama-fit-params` connect briefly before `llama-server`; they must finish first.

## Commands used during a run

| Phase | Commands |
|---|---|
| projection (Gate 2) | `GET_DEVICE_MEMORY`, `GET_ALIGNMENT`, `GET_MAX_SIZE`, `GET_ALLOC_SIZE` |
| load | `ALLOC_BUFFER` (refused by the worker reserve if memory would run low), `BUFFER_GET_BASE`, then per weight tensor `SET_TENSOR_HASH` (worker cache hit: nothing sent) or `SET_TENSOR` with the data (worker saves it to its cache with `-c`) |
| inference | `SET_TENSOR` / `SET_TENSOR_2D` (split inputs), `GRAPH_COMPUTE` (first time) / `GRAPH_RECOMPUTE` (cached graph), `GET_TENSOR` (outputs), `COPY_TENSOR` (copies within one worker), event commands for async execution |
| end | `FREE_BUFFER`; on disconnect the worker frees everything of that client |

Tensor data crosses the wire in its own type: weights in their GGUF quantization, activations as F32.

## Errors

- Client side: any failed send/receive aborts `llama-server` ("Remote RPC server crashed or returned malformed response").
- Worker side: an unsupported op or a CUDA error aborts `ggml-rpc-server` (the launcher restarts it). The worker validates every tensor region against its buffers before reading or writing.
- A refused `ALLOC_BUFFER` (reserve) appears on Stage A as `unable to allocate RPCn[...] buffer` and in the worker log as `refusing ... must stay free`.

## Security

The protocol has no authentication or encryption. Only allow the worker ports from 192.168.50.206 (firewall `-RemoteAddress`), never forward them on the router.
