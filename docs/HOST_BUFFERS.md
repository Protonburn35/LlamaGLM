# Locked GGML host buffers

`host_buffer_pool` in src/glm-stage supplies a standard CPU-layout GGML buffer type backed by dedicated verified native pages. CPU buffer operations, alignment and allocation-size queries delegate to the pinned GGML CPU implementation. Native page ownership is now separate from the 64 MiB expert-read limit, so graph/cache buffers may use larger explicit page budgets. Generic native allocations remain bounded by address-space checks and a 1 TiB API ceiling; that ceiling is not an admission or hardware-capacity claim. Per-expert reads/buffers retain their original 64 MiB limit.

The pool reserves page-rounded bytes and data-buffer count under one mutex before allocating. Reservations include pending operations, live buffers and buffers being freed; budget releases follow actual native-page deletion. Nonempty buffers retain the allocator implementation and its type/ledger after the C++ wrapper is destroyed. GGML itself creates zero-byte dummy buffers without invoking a type's allocation callback. These own no payload pages and borrow the pool/type lifetime; keep the pool alive while using such dummy buffers. Allocator/map and GGML metadata remain outside the payload page budget.

Allocation exceptions stop at the C callback and return null, with failure/native-error counters. Capacity refusal returns null without a heap fallback. An incomplete allocation releases its reservation and any acquired native pages. A failed allocation of a later buffer in a GGML context frees earlier buffers through the existing GGML cleanup path. `verify_all` checks native page locks only when no allocation/free work is pending; it holds the registry lock against buffer deletion. Callers must complete backend work before freeing tensor buffers.

The primitive checkpoint has four CPU suites: independent ordinary-versus-locked matrix results for all eight publisher quant types, actual GGML scheduler reserve/compute with every tensor in the requested type, shared capacity/page rounding/dummy/partial-context cleanup, and eight concurrent requests plus retained-buffer lifetime. All four pass locally with exact outputs, and the full gate now has eleven CTest targets plus six GLM tests. These four suites pass native Windows at 54265dd8b, run 39551995800.

Explicit preload policy 2 now routes ordinary B host tensor/cache/graph/output/bank allocations through this pool; its exact-source Windows qualification is pending. Host payload protection does not establish complete process or GPU residency. Further integration must cover every relevant allocation path, including CPU backend work buffers, network activation buffers and staging, while preserving measured physical GPU/WDDM requirements. Full laptop admission stays false.

The optional Linux harness `scripts/glm-host-buffer-tsan.sh` requires shared GGML libraries compiled with matching ThreadSanitizer flags and `GGML_OPENMP=OFF`. The native GGML pthread path has sanitizer-aware barriers. Mixing an instrumented test with the ordinary OpenMP library produced synchronization warnings in both reference and locked buffers and is not accepted as sanitizer qualification. No suppression or disabled native lock check is used. All four suites pass with the matching instrumented GGML build, exit 0 and no report. This qualifies the allocator/pthread test configuration, not OpenMP race freedom. The harness uses the same real-kernel lock wrapper as the storage sanitizer; ordinary native tests use normal OS APIs.

Build only the shared CPU GGML targets from the repository root, then run the harness:

```sh
cmake -S . -B /tmp/glm-ggml-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_GLM_RPC=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF \
  '-DCMAKE_C_FLAGS=-fsanitize=thread -O1 -g' \
  '-DCMAKE_CXX_FLAGS=-fsanitize=thread -O1 -g' \
  -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread
cmake --build /tmp/glm-ggml-tsan --target ggml ggml-cpu -j 4
bash scripts/glm-host-buffer-tsan.sh /tmp/glm-ggml-tsan /tmp/glm-host-buffer-tsan
```


## Shared external preload reservation

preload_external_pages reserves an exact native page count under the same mutex before invoking its startup-only loader. Pending external pages compete with all tensor allocations. The returned owner and a verifier remain in the allocator; the loader is never stored. Verification must check native locks and return the exact page count without reentering the allocator. The owner must not retain the allocator or any of its buffers. Failed/empty/mismatched loads destroy temporary owners before releasing their charge. Nonempty GGML buffers retain this external owner through final allocator destruction. External payload pages count against the page budget, separately from the GGML data-buffer count and requested-byte metric.

A fifth component suite verifies reservation visibility before loading, reentrancy refusal, partial/null/mismatched cleanup, combined capacity and retained external-owner lifetime. All five suites pass matching instrumented GGML/pthread TSan. Policy 2 now uses this primitive for actual B host weights, recurrent/MLA/index state, graph/output tensors and CPU banks; see PRELOADED_EXPERTS.md. Native Windows primitive qualification passes at 54265dd8b (four suites); the expanded fifth suite and model integration await their exact-source run. Full backend work/process/GPU admission remains false.


## CPU backend workspace

The optional registry entry ggml_backend_cpu_set_work_buffer_type has signature bool(ggml_backend_t, ggml_backend_buffer_type_t). It accepts a supported host type once on a fresh idle CPU backend, before any direct graph/plan operation. Keep the type owner alive until that backend and all plans are freed. Standard callers do not install it. Policy 2 requires it before output/state/graph initialization; unsupported CPU modules refuse.

Configured direct work_data and explicit graph-plan scratch allocate from the requested type, with no heap fallback. Scratch growth allocates the replacement before freeing prior storage, so the page budget includes both during replacement. Allocation failure returns ALLOC_FAILED and preserves old scratch. The sixth component suite verifies quantized direct/plan parity, shared caps, failed growth and reuse, invalid/late configuration and cleanup. All six suites pass matching instrumented GGML/pthread TSan. The small KDA/DSA model oracle runs one thread so actual flash-attention scratch fits native test quotas; peaks are 3,457,024 / 7,081,984 bytes. GPU scratch, metadata, threadpool/control structures and network buffers remain outside full admission. Native Windows qualification of this follow-up is pending.

Windows scratch diagnostic at a343f870d reports native error 1453 (working-set quota): the 1,049,408-byte quantized work buffer fits its shared cap but exceeds the test process lock quota. The current correction scopes/restores the same test-only 16/64 MiB working-set limits used by model fixtures; native locks and production failure behavior remain unchanged. Restoration failure fails the fixture. Exact-source qualification is pending.

CPU scratch, explicit plans, test quota restoration and startup progress/cancellation now pass native Windows CPU in all three variants at 0fae41d97, run 39558328891 (eleven CTest, seven GLM). CPU artifact 12326731252 confirms six host suites and ten v1/v2 cancellation/reload cases. Production lock refusal is unchanged, and full working-set/GPU admission stays false. See RESIDENCY_STATUS.md for remaining allocation coverage.

A third local three-process profile now joins streamed A/C to policy-2 B under a separate small 2/1/3 CPU oracle. The harness removes B's same-basename GGUF copy before A connects. All 131 tokens have exact boundaries/logits; B retains all 48 expert slabs, unchanged startup reads, tracked native host locks and a 3,457,024-byte shared peak. Original profiles and all startup refusals remain. Complete working-set admission stays false; this expanded fixture awaits native Windows qualification.
