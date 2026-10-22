# Engineering Lessons from DeepSeekRPC Failure (Read-Only Reference)

## 1. Context
The DeepSeekRPC project was abandoned after failing its first real inference request with HTTP 500 (`stage_operation_failed`) and zero emitted tokens, despite all synthetic CUDA checks passing and `/readyz` returning `ready=True`. LlamaGLM replaces the Python/Triton/vLLM stack with native C++ `llama.cpp` + `ggml-cuda` + `ggml-rpc`, but incorporates the operational and architectural lessons learned.

## 2. Key Failure Modes & Concrete Regressions in LlamaGLM

1. **Do not confuse readiness with inference:**
   - DeepSeekRPC considered three running listeners and a handshake as "ready", but the actual execution graph failed on first token generation.
   - *LlamaGLM requirement:* The acceptance gate is a verified end-to-end generated token from real weights across all three PCs, not just a successful `/readyz`.

2. **Accurate and actionable error diagnostics:**
   - DeepSeekRPC returned generic `stage_operation_failed` HTTP 500 responses without tracebacks.
   - *LlamaGLM requirement:* Detailed stage, layer, and operation logging with explicit error messages and exception logging in server output.

3. **Robust cancellation and rollback:**
   - In DeepSeekRPC, client request cancellation resulted in `InterruptedError` in background loops that poisoned subsequent admissions.
   - *LlamaGLM requirement:* Cancellation cleanly resets KV cache slots and returns the server to an uncorrupted ready state without killing the worker loop.

4. **WDDM and local memory headroom:**
   - Rigid 1GB headroom assertions crashed Stage B on Windows WDDM even when physical resources were sufficient.
   - *LlamaGLM requirement:* Configurable VRAM headroom reserves and graceful fallback to bounded host-RAM / SSD expert paging.

5. **No rigid code fingerprints:**
   - DeepSeekRPC's launcher rejected minor debug edits with obscure "Run Install.ps1" errors.
   - *LlamaGLM requirement:* Development mode support and clear binary verification without brittle launcher lockouts.
