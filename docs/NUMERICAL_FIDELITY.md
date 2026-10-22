# Numerical fidelity

- **Weights:** used exactly as stored in `UD-IQ3_XXS` (mixed IQ2_S / IQ3_S / IQ4_XS / Q6_K / ... per tensor). Uploads to the workers send the GGUF bytes unchanged. CPU weights on Stage A are read through mmap. No re-quantization anywhere.
- **Experts:** all 8 routed experts plus the shared expert are computed for every token in every MoE layer, on whichever device holds that layer. There is no expert dropping, caching substitute or paging on the workers.
- **Activations between devices:** `ggml_backend_sched` copies tensors in their own type. The residual stream crossing a device boundary is F32: 4 hyper-connection streams x 4096 x 4 bytes = 64 KiB per token (32 MiB per 512-token micro-batch).
- **KV / state:** default F16 KV for the DSA (MLA) cache; KDA recurrent state F32. No KV quantization flags are set.
- **Precision risk found in the cross build:** Gemini's clang flags compiled the CUDA kernels with `-ffinite-math-only` (via `-ffast-math`), which makes the `-INFINITY` sentinels used by softmax, masks, flash attention and the MoE top-k router undefined behaviour. Fixed in engine commit `502ff59` (`-fno-finite-math-only`). nvcc's `-use_fast_math` (upstream builds) never had this problem.
- **Cross-check available:** the same plan can run on the official nvcc build of the same source (`-BinDir`, `FIRST-RUN.md` section 6). With `temperature = 0` both builds should give very similar, sensible text; exact token equality is not guaranteed because kernels and reduction order can differ.
- **Not verified on hardware:** perplexity or output quality versus a reference.
