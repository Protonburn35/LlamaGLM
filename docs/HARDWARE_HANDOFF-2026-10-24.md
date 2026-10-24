# Hardware results and optimization handoff (2026-10-24)

## Current request

The user has finished hardware testing for today. They will send Claude `Llamaglm logs.zip` separately after this handoff is published. Codex reviewed that archive; raw logs and private machine data are not committed. Use the filenames below to find the evidence when the archive arrives.

The requested baseline is **MTP off**, using the working 29-layer offload plan below. The last successful archived run has MTP on and a different 28-layer plan; a later restart into the requested baseline is not captured. The user also wants Claude to consider an **optional experimental MTP setting** for a future build and explicitly evaluate **expert caching**. Keep a baseline without speculation and provide experiments as opt-in options.

Claude should decide which optimizations are worth implementing after reviewing the evidence and implementation costs. The user is open to several experiments in the next build if they are optional, reversible, and preserve the model's capabilities and normal sampling behavior. The candidates below are for assessment, not an instruction to implement everything.

Requirements for the next work:

- Do not treat network bandwidth as limiting performance until measurements show that it does. Higher network usage is welcome when it improves speed. RPC latency is a separate hypothesis that also needs timing evidence.
- Preserve the same model, mixed tensor quantization, all selected experts, and target sampling behavior. Do not gain speed by dropping experts, changing the quantization, silently reducing context, or disabling reasoning. Normal nonzero temperatures must work; temperature 0 must not be required.
- Preserve RPC correctness tests, memory-fit checks, and worker reserves. The user accepts skipping repeated local CUDA self-tests on the same tested build; they specifically prefer keeping the RPC test enabled.
- Keep experimental behavior behind explicit options with documented defaults and rollback. Measure loading, prefill, generation, and memory costs separately.
- Both Claude and Codex may commit and push to `master`; fetch first and preserve each other's commits. CI cleanup remains deferred, with AMD/Vulkan coverage to retain. Follow [CI_SCOPE.md](CI_SCOPE.md) to avoid another inherited-workflow storm.

## Logged results

The package identifies itself as build 1, commit `bb0bc7d`, MSVC 19.44.35229.0, Windows AMD64. All three GPUs report driver 621.01. The model is the four-shard Unsloth GLM-5.3-Flash `UD-IQ3_XXS`, with 45 trunk blocks and one NextN block. Tested settings: 8K context, one slot, logical batch 2048, microbatch 512, `--no-repack`, and `--no-op-offload`. The user's API is `127.0.0.1:8081`; the original repository default is 8082.

Hardware visible in the logs:

| PC | CPU | GPU | Configured inference threads |
|---|---|---|---|
| A | Ryzen 7 7800X3D | RTX 4090, 24 GiB | 8 |
| B | Ryzen AI 9 HX 370 | RTX 4060 Laptop, 8 GiB | 12 |
| C | Ryzen 7 9800X3D | RTX 5070 Ti, 16 GiB | 8 |

Each PC has approximately 32 GiB RAM. B is an AMD laptop; do not apply the older documentation's hypothetical Intel P/E-core assumptions to this machine.

CUDA self-tests completed on all three GPUs: the reported groups passed 420/420 matrix tests, 24/24 flash-attention tests, and 1642/1642 other operation tests per GPU. Physical-worker Gate 1 also passed. These are real hardware results, distinct from hosted CPU fixtures. The short `stage-c-20261024-133200.log` stops during self-test startup; `stage-c-20261024-133213.log` contains the completed tests and worker session. The short file alone does not establish a CUDA crash.

Two full-model requests completed:

| Evidence file | Mode and placement | Prompt processing | Generation | Other evidence |
|---|---|---|---|---|
| `llama-server-20261024-134750.log` | MTP off; `ngl=29`, `ts=7,7,2,7,6` | 214 tokens in 62,609.85 ms; **3.42 tok/s** | 518 reported tokens in 303,157.28 ms; **1.71 tok/s** | Model loaded about 10 min 48 s after server startup; zero graph reallocations |
| `llama-server-20261024-141606.log` | `draft-mtp`, `n_max=1`; `ngl=28`, `ts=7,7,2,7,5` | 214 tokens in 58,915.50 ms; **3.63 tok/s** | 411 reported tokens in 215,514.86 ms; **1.90 tok/s** | Model loaded about 4 min 40 s after server startup; zero graph reallocations; **184/227 drafts accepted (81.057%)**, mean accepted length 1.81 |

In the original files, baseline sampler settings are at line 359 and final timings at 444-448; MTP settings are at 347-349, sampler settings at 376, and final timings/acceptance at 443-450. Both used `temp=1.000`, `top_k=40`, `top_p=0.950`, and `min_p=0.050`. MTP actually executed at nonzero temperature. Request completion and passing operation tests do not establish broad answer quality, perplexity, or distribution-preserving speculative sampling; those need appropriate validation.

The earlier conversational estimate of approximately 200 tok/s prefill is **not supported by this archive**. The completed requests show about 3.4-3.6 tok/s; the prompts contained 214 tokens. Use the logged timing fields as the baseline.

The reported MTP generation rate is approximately 11% higher, but this is **not a controlled measurement of its benefit**: placement changed, output lengths differed, cache/load conditions differed, and equal prompt token counts do not establish identical request content. The user observed no noticeable improvement. Do not call MTP broken, promise a speedup, or equate high draft acceptance with high generation speed.

Worker logs show CPU and CUDA graph execution and startup disk-cache reuse. The final MTP session reports RPC-managed buffer peaks around 5.71 GiB GPU / 18.47 GiB CPU on B and 11.81 GiB GPU / 18.55 GiB CPU on C, with zero refused worker allocations in those summaries. These are buffer-accounting figures, not total GPU memory usage or execution timings. Graph/node counts do not identify the slowest device.

## Placements and fit failures

The `devices` order was `CUDA0,RPC1,RPC0,RPC3,RPC2`: A GPU, B RAM, B GPU, C RAM, C GPU. Global RPC0/RPC1 are B GPU/RAM; global RPC2/RPC3 are C GPU/RAM. Worker-local buffer names reuse RPC0/RPC1 per endpoint; read the endpoint as well as the device label.

| Configuration | `ngl` | `ts` | Result |
|---|---|---|---|
| Shipped balanced plan | 31 | `7,8,2,8,6` | Initial Gate 2 refused worker RAM: B short 1709 MiB, C short 403 MiB after margins |
| Working baseline, MTP off | 29 | `7,7,2,7,6` | Gate 2 passed and a request completed |
| MTP attempt on that split | 29 | `7,7,2,7,6` | Launcher Gate 2 passed; server fit check then refused C GPU before loading |
| Working MTP trial | 28 | `7,7,2,7,5` | MTP initialized and a request completed |

All used `fit_margin_mib=1536,2560,512,2560,1024`; worker reserves remained B `384,2048` and C `512,2048` MiB. No reserve reduction was needed. The initial failure is in `stage-a-20261024-133840.log`; passing baseline checks are in `stage-a-20261024-134439.log` and `stage-a-20261024-134711.log`.

Actual engine assignments, rather than stale descriptive `layers` strings in the JSON:

| Pool | Baseline, MTP off | MTP trial |
|---|---|---|
| A CPU/mmap | 0-17 | 0-18 |
| A 4090 | 18-24 | 19-25 |
| B RAM | 25-31 | 26-32 |
| B 4060 Laptop | 32-33 | 33-34 |
| C RAM | 34-40 | 35-41 |
| C 5070 Ti | 41-44 and output; block 45 skipped | 42-44, NextN block 45, and output |

Reducing C's GPU count shifted the boundaries throughout the split, kept both workers at seven RAM layers, and added one CPU/mmap layer on A. A's mapped weight inventory increased from about 40.21 GiB to 42.75 GiB. A mapping larger than free RAM is not proof that decode is SSD-bound.

The MTP fit discrepancy is concrete: `stage-a-20261024-141138.log` passed the 29-layer launcher projection, but `llama-server-20261024-141215.log` line 27 reports C GPU **14903 MiB free, 14520 MiB needed, plus 1024 MiB margin**, then refuses to load. Environment speculation options are registered for the server, not the standalone fit tool's example category; the printed CLI also omits inherited environment values. An experimental MTP option must use matching effective settings for projection and startup and account for both target and draft contexts. Existing server checks caught this attempt. It was a fit refusal, not an allocation crash.

## User observations and the Strata reference

These observations were reported by the user, not captured as hardware-counter traces in this ZIP:

- The 4090 PC's SSD was not being hit hard during output. Prefill caused some reads. The old assumption that SSD paging explains the current generation rate needs measurement.
- Output network traffic peaked around 25 Mbps. Prefill used more bandwidth but did not appear to saturate it. Neither phase has an established bandwidth bottleneck.
- StrataRPC ran the **same model and quantization on A and C, without the laptop**, and was substantially faster, with observed traffic around 667 Mbps. Exact output tok/s, revision, configuration, and MTP state are not supplied. Obtain/reproduce those details when available; do not substitute a different-model comparison.
- A and the laptop are easy to monitor; C is upstairs. Optional centralized telemetry/log collection would simplify the next session.

Both engine/placement and node participation changed in the Strata comparison. Adding B helps memory capacity but may increase latency; measure its contribution. A two-desktop profile needs a new fit/placement plan, not deletion of an endpoint from this JSON.

Higher traffic can result from producing more tokens per second. Compare bytes per generated token as well as bytes per second. Do not infer Strata's execution strategy or a present RPC-latency bottleneck from traffic volume. Historical `LIMITATIONS.md` estimates are not measured hardware ceilings, and assumed bandwidth limits must not rule out potentially useful implementations.

## Candidates for Claude to assess

### Profiling and reproducible experiments

Add optional low-overhead summaries of CPU/GPU work, RPC command/transfer counts and bytes, waits, and per-device/stage elapsed time, separately for prefill and decode. Distinguish GPU enqueue time from completed work; avoid adding frequent synchronization that creates a new bottleneck. Existing 30-second buffer/graph statistics cannot attribute execution time. Record request/output lengths, sampling, placement, threading, cache/MTP settings, and warm/cold load state with each result.

The completed runs already report six graph splits and zero graph reallocations. Measure graph/RPC costs before assuming rebuild churn is responsible. Include peak real RAM/VRAM and page-file/file-read activity, not just idle free memory. A lightweight view of C from A is useful; do not create a second active RPC client contending with inference.

### Expert caching and tensor placement

**Explicitly assess expert caching.** Quiet SSD activity does not rule out gains from moving repeated CPU/RAM expert work onto a GPU. Measure the time saved and refill costs, not only hit rates.

The engine already has an adaptive SLRU cache in `src/llama-moe-cache.cpp`, controlled by `--moe-cache-mib` / `LLAMA_ARG_MOE_CACHE_MIB`, default off. It needs GPU-assigned layers with routed expert weights in local host weight buffers. The current whole-layer CPU/GPU placement has no eligible layers; setting the flag alone is not a useful trial. Misses upload experts and compute on the GPU. This is not the proposed CPU-miss/GPU-hit hybrid.

A bounded first candidate is a **4090-only profile**: make a small number of its CPU layers GPU-assigned, retain those layers' routed expert banks in host memory, and keep other tensors on the GPU. Start near 1 GiB of cache only if measured peaks leave room; initially preserve the existing fully GPU-resident expert banks. Verify actual operation placement with `--no-op-offload` rather than equating layer assignment with expert execution.

Compare the current baseline, the changed placement with cache off, and that same placement with cache on. Record byte-weighted hits/misses, evictions/refills, CPU/GPU time, PCIe traffic, peak memory, and generation speed. Include large prefill and MTP verification batches: their selected-expert union can exceed available slots and trigger fallback. Preserve all eight selected experts and original tensor types; do not replace the model with a fixed subset of experts.

Account separately for unique weights, host backing, cache copies, context/recurrent state, compute buffers, and MTP state. Moving entire resident banks to RAM may exhaust RAM. Each configured `llama_context` constructs a cache, so audit possible duplicate cache allocations in target/MTP contexts sharing one model before combining those options.

Longer term, use independent worker-local budgets, including zero, according to measured benefit. B's GPU was tight: the baseline shutdown snapshot reported only 561 MiB free. Remote RAM is not host-addressable from A; do not mark RPC RAM as local host memory to bypass cache eligibility.

Before designing a distributed engine, inspect the existing same-server `COPY_TENSOR` path in `ggml/src/ggml-rpc/ggml-rpc.cpp`. It delegates to `ggml_backend_buffer_copy_tensor`, while CUDA's destination callback currently accepts CUDA sources; a CPU-to-CUDA copy may fail that fast path and fall back through A. Verify and, if worthwhile, extend worker-local copies so expert payloads can remain between a worker's RAM and GPU. Reuse cache policy and kernels where practical.

Also assess placing attention, routers, shared experts, and other weights needed every token on each owning GPU while retaining routed experts in RAM. This changes graph splits/transfers as well as CPU work, so benchmark it. Available network capacity may be used to improve speed. A CPU-miss/GPU-hit hybrid is separate implementation work, not a feature already enabled here.

### Optional experimental MTP

Expose the **existing, hardware-executed** `draft-mtp` implementation through an explicit optional launcher/config feature, baseline off. New JSON keys are for Claude to design; an arbitrary `mtp` field in today's JSON has no effect. The successful trial used `LLAMA_ARG_SPEC_TYPE=draft-mtp` and `LLAMA_ARG_SPEC_DRAFT_N_MAX=1`. No second model download was required.

The NextN block has about 2.608 GiB of weights, placed on C here, not that amount on every PC. Budget the block, target rollback state, draft context, compute buffers, and any duplicated caches. Make fit/startup settings agree, print effective draft length/acceptance, and give a device-specific refusal when the option does not fit. Preserve user sampling and validate nonzero-temperature behavior.

Benchmark off/on with the same placement and comparable requests, then combine with a promising cache profile as a separate experiment. Measure delivered tok/s, draft/verification work, acceptance, and peak memory. Acceptance alone is not a speed metric. Keep the option experimental until memory and sampling behavior are validated; the supplied run proves execution, not broad quality preservation.

The successful run's speculative summary reports 4300.711 ms of draft-generation time across 227 draft calls, while the request's generation phase took 215514.86 ms. Confirm those timer semantics and measure target verification separately; the logs do not justify assuming drafting is the largest cost merely because MTP is enabled.

### Other candidates, selected by measured benefit

- CPU tuning: examine backend time, B's AMD laptop thread selection, and optional CPU variants for the desktop instruction sets. Validate actual kernel support, package/DLL discovery, memory use, and speed. More threads or wider SIMD are hypotheses to test.
- Prefill: the measured approximately 3.5 tok/s makes it an optimization target too. Assess batching and the existing `-OpOffload` option with peak-memory accounting; do not silently enable whole-bank transfers or reduce context/sampling quality to claim a gain.
- Scheduling/transport: assess worker-local execution, independent CPU/GPU expert work, useful compute/copy overlap, and fewer coordinator-mediated transfers where profiling justifies them. Autoregressive layer dependencies do not rule out concurrency within a layer or better stage execution.
- Parallelism: distinguish one-chat latency/tok/s from aggregate multi-request throughput and prefill microbatch pipelines. Evaluate tensor parallelism using real timings/traffic on these PCs, without rejecting it on a presumed bandwidth ceiling.
- Node participation: weigh B's capacity benefit against its measured per-token cost and examine the working two-desktop Strata setup. Do not make recreating the abandoned DeepSeekRPC project a prerequisite for smaller improvements.

## Validation, collection, and next delivery

Keep a known working baseline and isolate each experiment's effects even if several optional features ship together. Reuse existing CPU/RPC fixtures, operation comparisons, reserve regression, argument/launcher checks, and the manual hosted Windows package workflow as appropriate. Full weights are on the user's PCs; do not download the full model into cloud/CI or require another hardware session today. Prepare a build, a small test matrix, option descriptions, and rollback instructions for the next session.

Validate cache fill/eviction/fallback, all selected experts, mixed quantization, and target-verified speculation at normal sampling temperatures. Deterministic seeds or greedy runs may help internal checks, but the user must not need temperature 0, expert clipping, or reduced reasoning. Preserve the baseline's capabilities and sampling contract; tiny tests alone do not establish full-model answer quality.

The ZIP contains 14 `.log` files from A/B/C, but no `system-*.txt`, current JSON snapshot, continuous utilization trace, or Strata benchmark. Effective CLI in stage logs supplies placement and API port; reconstruct inherited options only where server evidence supports them. Do not claim unrecorded settings or per-stage timings.

For future Windows command blocks, include `cd C:\LlamaGLM` first. A manually opened PowerShell running `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ...` preserves readable errors when scripts finish. Retain RPC tests; allow `-SkipSelfTest` for repeats on the same build. Workers can stay running across coordinator restarts.

Logging already starts with the launchers; `collect-logs.ps1` packages it afterward. The collector copies up to ten logs per readable `\\<worker>\LlamaGLM-Logs` share, but creates a system snapshot only on the PC where it runs. Consider capturing effective config/relevant option values in future bundles. The user will send this archive directly to Claude; do not publish raw logs or assume share collection also captured remote hardware snapshots.
