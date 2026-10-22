# Build log

Imported VM report for the original v0.8.0 snapshot. Independently repeated checks of the GitHub import are listed in [REVIEW-v0.8.0.md](REVIEW-v0.8.0.md).

## Toolchain (Linux VM -> native Windows x64)

- Host C/C++ and final link: MinGW-w64 `x86_64-w64-mingw32-g++-posix` 13 (`toolchain-mingw-cuda-win64.cmake`).
- CUDA device + host code of `.cu` files: clang 20.1.2 targeting `x86_64-w64-mingw32` through `cuda-clang-wrap.sh` (clang for `-c`, mingw g++ for links). Device code assembled by the Linux CUDA 12.8 `ptxas`/`fatbinary`, linked against the Windows import libraries from the CUDA 12.8 redist in `~/projects/cuda128/cross`.
- Architectures: `CMAKE_CUDA_ARCHITECTURES=89;120`; the generated commands use `--cuda-gpu-arch=sm_89 --cuda-gpu-arch=sm_120a`.
- This is **not** the upstream-supported way to build ggml-cuda (upstream uses nvcc). The official nvcc/MSVC build of the same source is llama.cpp release b11780, which can be used as a reference (`FIRST-RUN.md` section 6).

### Environment pieces outside the engine repo (needed to reproduce the build)

1. **Patched system header** `/usr/lib/llvm-20/lib/clang/20/include/__clang_cuda_cmath.h` (package `libclang-common-20-dev 1:20.1.2-0ubuntu1~24.04.3`; `dpkg -V` reports it modified). Two lines wrap clang's device `isinf/isfinite/isnan` overloads in `#if !defined(CLANG_CUDA_CMATH_SKIP)`; the toolchain passes `-DCLANG_CUDA_CMATH_SKIP` because those overloads clash with mingw's `<cmath>`. The exact diff is saved in `toolchain/clang-20-cuda-cmath-skip.patch`. To restore after a package reinstall:
   ```bash
   sudo patch -p0 -d / < toolchain/clang-20-cuda-cmath-skip.patch   # paths in the patch are relative to /
   ```
   With the macro, device `isinf()`/`isfinite()` resolve to clang builtins; they compile to real compares as long as finite-math is off (checked in PTX).
2. `~/projects/cuda128/host/root/include/clang-cuda-shim` (copy in `toolchain/clang-cuda-shim`): device placement `new`/`delete` for CCCL; included with `-include clang-cuda-shim`.

### CPU backend SIMD (fixed in the build options)

`ggml/CMakeLists.txt` sets `GGML_NATIVE_DEFAULT=OFF` when `CMAKE_CROSSCOMPILING`, and then all x86 options (`GGML_SSE42/AVX/AVX2/FMA/F16C/BMI2`) default to OFF. Gemini's `build-win-cuda` was configured that way, so `ggml-cpu.dll` contains only scalar code. Required options now: `-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON`; `package-release.sh` checks `GGML_AVX2:BOOL=ON` in the cache.

### Device math flags (fixed in engine commit 502ff59)

Gemini's flags: `-Xarch_device -ffast-math`. clang's `-ffast-math` includes `-ffinite-math-only`; ggml-cuda uses `-INFINITY` in softmax, masks, flash attention, DSV4 hyper-connections and the fused MoE top-k router, so those kernels were compiled with undefined behaviour (206 warnings in `topk-moe.cu`, 47 in `softmax.cu`, different PTX). Now: `-Xarch_device -ffast-math -Xarch_device -fno-finite-math-only`, which keeps the approximations nvcc's `-use_fast_math` allows and IEEE inf/nan semantics. After a build, `grep -c Wnan-infinity-disabled <build log>` must be 0.

## Builds

| Date | Source | What | Result |
|---|---|---|---|
| 2026-10-21 | `5453314` (Gemini) | Windows CUDA build -> `dist/LlamaGLM-v0.7.0-*.zip` | built; **defective** (finite-math device code, RPC mmap off). Do not use. |
| 2026-10-21 | `5453314` | Linux CPU build (baseline for tests) | OK |
| 2026-10-21 | `502ff59` | Linux CPU build (`ggml-rpc-server llama-server llama-completion test-llama-archs llama-fit-params llama-rpc-test test-backend-ops`) | OK, no errors |
| 2026-10-22 | `502ff59` | **Windows CUDA build v0.8.0** (Claude), commands of `NEXT_AGENT_START_HERE.md`, `ninja -C build-win-cuda` (all targets) | **OK**: 376 steps, 15 min 40 s on 8 vCPU, exit 0, 0 errors, **0 `-Wnan-infinity-disabled`**. Checks below. Not run on a GPU. |

### v0.8.0 build checks (2026-10-22)

- Configure: `GGML_SSE42/AVX/AVX2/FMA/F16C/BMI2 = ON` in `CMakeCache.txt`; CMake reports `Adding CPU backend variant ggml-cpu: -msse4.2;-mf16c;-mfma;-mbmi2;-mavx;-mavx2`. `-fno-finite-math-only` appears 145 times in `build.ninja`. CUDA archs `89;120a`.
- `common/build-info.cpp`: `LLAMA_COMMIT = "502ff59"` = engine HEAD.
- Warnings (log `build-v0.8.0.log`, not in git): 7148 `-Wpass-failed=transform-warning` ("loop not unrolled", 7120 of them at `common.cuh:441`, the `ggml_cuda_unroll` helper used by the flash-attention and MMQ templates: clang could not apply some `#pragma unroll`; affects speed, not results), 435 `-Wunknown-warning-option` and 145 `-Wunused-command-line-argument` (nvcc-style flags passed to clang), 435 `-Winvalid-noreturn` and 214 `-Wreturn-type` (host side of device-only functions, upstream code), 145 `-Wunknown-cuda-version` (clang 20 marks CUDA 12.8 as partially supported), 28+28 in vendored `subprocess.h`, 8 `-Wformat=`.
- `ggml-cpu.dll`: 2706 AVX2/FMA instructions (`vpmaddubsw`, `vfmadd*ps`, `vpmaddwd ymm`), 19681 `ymm` uses (v0.7.0: 0).
- `ggml-cuda.dll` (336 MB): 141 fatbins, each with an sm_89 and an sm_120a ELF image (parsed from the fatbin headers), no PTX. Imports only `cudart64_12.dll`, `cublas64_12.dll`, `nvcuda.dll`, `ggml-base.dll`, the MinGW runtime and `KERNEL32`/`msvcrt`.
- Third-party DLLs in `bin/` (copied there on 2026-10-21) are byte-identical to `~/projects/cuda128/win/root/bin` (CUDA 12.8 redist) and to the MinGW runtime DLLs of the toolchain. Kept in the package: `cublasLt64_12.dll` references `nvrtc64_120_0.dll` by name and may load it at run time.

## Package contents (v0.8.0, `scripts/package-release.sh`)

`llama-server.exe llama-cli.exe llama-bench.exe llama-fit-params.exe ggml-rpc-server.exe llama-rpc-test.exe test-backend-ops.exe llama-gguf-split.exe`, all DLLs of the build, MinGW runtime DLLs, `cudart64_12.dll cublas64_12.dll cublasLt64_12.dll`, the launchers and `cluster_config.json`, docs, `MANIFEST.txt` (engine and launcher commits + SHA-256 of every file). Also `LlamaGLM-v<version>-source-<engine commit>-<launcher commit>.zip` (both repos; the engine is in `LlamaGLM-master/`) and a `SHA256SUMS` file. The script refuses to run if either repo has uncommitted changes.
