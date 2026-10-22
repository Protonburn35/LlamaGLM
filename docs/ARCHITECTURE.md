# Architecture

LlamaGLM runs GLM-5.3-Flash (`unsloth/GLM-5.3-Flash-GGUF`, `UD-IQ3_XXS`, 112.1 GiB) on three Windows 11 PCs with stock llama.cpp mechanisms: one `llama-server` process on Stage A drives everything; Stages B and C run `ggml-rpc-server`, which only allocates buffers and executes the graph pieces it is sent. There is no custom pipeline engine.

```
Stage A 192.168.50.206 (RTX 4090 24 GB, 32 GB RAM)       Stage B 192.168.50.129:50052        Stage C 192.168.50.143:50053
llama-server  http://127.0.0.1:8082                       ggml-rpc-server -d CUDA0,CPU        ggml-rpc-server -d CUDA0,CPU
  model files (mmap)                                        RPC0 = RTX 4060 Laptop 8 GB         RPC2 = RTX 5070 Ti 16 GB
  token_embd + layers 0-15 on CPU (mmap)                    RPC1 = laptop RAM                   RPC3 = RAM
  layers 16-22 on CUDA0                                     layers 23-30 on RAM (RPC1)          layers 33-40 on RAM (RPC3)
  tokenizer, sampling, HTTP API                             layers 31-32 on GPU (RPC0)          layers 41-44 + output on GPU (RPC2)
        |  weights at load (once, then cached by hash on the workers)  |
        |  per graph split: input activations in, results out (TCP)   |
        +--------------------------------------------------------------+
```

(`balanced` plan; `docs/MEMORY_PLAN.md` has the numbers and the `laptop-light` alternative.)

## How a token is computed

`ggml_backend_sched` splits each forward graph by the device that holds the weights of each op: A CPU -> A CUDA0 -> B RAM -> B GPU -> C RAM -> C GPU. At every device change it copies the split's inputs (the 4-stream hyper-connection residual, F32: 4 x 4096 x 4 B = 64 KiB per token, plus small tensors) to the next device. For RPC devices that is a `SET_TENSOR` over TCP to the worker; the worker then runs the split on its CUDA or CPU backend (`GRAPH_COMPUTE`). Logits come back from C's GPU to A, where sampling happens. Copies keep the tensor type (F32 activations stay F32). All 8 routed experts of every MoE layer are computed; nothing is dropped or re-quantized.

Two transitions are inside one worker (B RAM <-> B GPU, C RAM <-> C GPU) and are done by the worker itself (`COPY_TENSOR`). The others cross the LAN; a copy from B to C goes through Stage A (read from B, write to C), so it costs two LAN transfers.

## Memory safety layers

1. `cluster_config.json`: plan with exact per-device slot counts (`sum(ts) == ngl` is checked).
2. Gate 2 / `--fit-check`: before loading, a no-allocation projection (weights + KV/state + compute) per device is compared with that device's free memory minus a margin; the load is refused if anything does not fit. Stage A's CPU weights are mmapped and not counted (the OS can drop and re-read those pages).
3. Worker reserve (`GGML_RPC_SERVER_RESERVE_MIB`): each worker refuses any allocation that would leave less than its reserve free on its GPU or in its RAM.
4. Windows driver setting "CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback": a full GPU fails fast instead of spilling into shared RAM.

## Observability

- Stage A log: `load_tensors: layers A-B (... recurrent) -> DEVICE (endpoint)` per range, `... model buffer size` per device, `common_fit_check` table, `load_mode = mmap`.
- Worker log: `[stats] device N (CUDA0|CPU): buffers, received, from cache, sent, graphs (nodes)` every 30 s of activity and at disconnect.
- `smoke-test.ps1` checks the Stage A log and real token counts; `collect-logs.ps1` gathers logs and system state.

## Failure behaviour (upstream)

- A worker that dies (CUDA error, crash) makes `llama-server` abort with "Remote RPC server crashed or returned malformed response". The worker launcher restarts the worker (3 times); Stage A is then started again.
- Stage A can be restarted at any time; workers free that session's buffers and wait for the next connection.
- There is no request-level recovery or cancellation forwarding to workers.
