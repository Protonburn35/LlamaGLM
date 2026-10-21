# Middle-stage working-set coverage

Full laptop residency/admission is **NOT_VALIDATED**. Experimental preload policy 2 is an allocation policy, not permission for a production worker to send READY. It shares one page-rounded host budget across the complete expert catalog and the GGML host allocations below. Native allocation/lock failure refuses; there is no pageable data fallback. The configured API cap is not measured physical capacity.

| Allocation | Current policy 2 coverage | Remaining work |
| --- | --- | --- |
| Complete owned routed expert payloads | Startup-only reads into verified locked pages; immutable memory-only inference reader; zero retained expert files; charged before startup reads | Real mixed-quant model and physical laptop qualification |
| Ordinary owned host weights | Locked standard CPU-layout buffers under the shared budget; extra packing/overrides refused | Device weights need separate physical GPU admission |
| Host KDA, MLA and index tensor state | Locked GGML buffers; FP32 KDA and FP16 MLA/index preserved | Full target context and device-state qualification |
| CPU expert banks | Locked raw-quant bank buffers; padded allocation included in shared and per-context caps | GPU banks retain device buffers and need physical residency checks |
| CPU scheduler graph tensors | Locked host buffer type; actual graph-node payloads checked by independent CPU fixtures | GPU graph/workspace accounting and worst-case production reserve |
| Host outputs and residuals | Locked buffers through the same policy; all four FP32 streams preserved | Production queue ownership, staging and cancellation lifetimes |
| CPU backend work_data and explicit-plan scratch | Optional fresh-backend capability selects the locked pool; bounded growth retains old scratch until replacement succeeds | Passes hosted native Windows CPU at 0fae41d97; production reserve/physical qualification remains |
| GGML descriptors/contexts, graph metadata, tokenizer/config maps and allocator records | Ordinary heap/control allocations | Dedicated ownership, bounded accounting and native protection |
| CPU/OpenMP threadpool/control data and stacks | Ordinary backend/runtime allocations | Bound/account and protect the inference working set |
| Authenticated wire frames and reassembled activations | Protocol limits, authenticated completion and lifetime checks; ordinary vectors | Shared host budget, native protection and production peer/queue quotas |
| CUDA transfer staging | Ordinary backend/driver behavior; OS-locked payloads are not CUDA DMA registration | Explicit bounded transfer storage, lifetimes and admission |
| GPU allocations and Windows/WDDM resources | Logical backend bank counters and configured limits only | Actual devices, measured available/reserved resources and residency validation |
| OS, other processes and operating reserve | No complete physical admission | Capacity probes, explicit reserve and fail-before-READY policy |

The host report verifies only tracked native payload pages, between decodes and allocation/free operations. Its full_residency_validated field remains zero. Its tensor buffer/byte counters also include configured CPU work buffers. No production negotiation should infer full admission from these counters, source/header identity or successful hosted compilation.

Policy 1 protects only the complete expert payload catalog. The existing three-process fixture still uses buffered B, including its desktop-streamed profile; that fixture does not qualify this complete host policy across processes. Production workers/READY, distributed transactions and all-owner reset remain absent.

All actual model inference so far uses small synthetic F32 CPU oracles. Independent component operators cover the eight publisher quant types. The target real mixed-quant model, GPU runtime, physical three PCs and 1,048,576-token execution remain NOT_RUN. See PRELOADED_EXPERTS.md, HOST_BUFFERS.md, TEST_REPORT.md and BUILD_LOG.md for exact source evidence and fixture limits.
