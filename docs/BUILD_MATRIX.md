# Build matrix

| Configuration | Source | Compilers | Status |
|---|---|---|---|
| win-cuda-cu12.8-x64 v0.7.0 | `5453314` | MinGW g++ 13 + clang 20 (CUDA) | built by Gemini; **defective, do not use** (RPC mmap off, finite-math CUDA kernels, scalar CPU backend) |
| win-cuda-cu12.8-x64 v0.8.0 | `502ff59` | MinGW g++ 13 + clang 20 (CUDA), `sm_89` + `sm_120a`, AVX2 CPU backend | **built 2026-10-22**, Wine CPU test OK (TEST_REPORT T20, T21); not run on a GPU |
| win-cuda-cu12.8-x64 v0.8.1 | packaged source `bb0bc7d`, exact commit in `MANIFEST.txt` | MSVC 2022 + nvcc 12.8.1 on GitHub Actions `windows-2022`, `89-real;120a-real`, AVX2 CPU backend, app-local MSVC runtime | package checks passed; 2026-10-24 user logs confirm self-tests on all three GPUs, physical-worker Gate 1, and completed full-model requests with MTP off/on; see [hardware handoff](HARDWARE_HANDOFF-2026-10-24.md) |
| linux-cpu (test rig) | `502ff59` | gcc 13.3 | built, tests in `TEST_REPORT.md` |
| official llama.cpp b11780 win-cuda-13.4-x64 | `1242be66f` | MSVC + nvcc (upstream CI) | reference build for `-BinDir`; needs driver R580+ |

v0.8.1 build options: see `.github/workflows/llamaglm-package-windows.yml`.

v0.8.0 (clang cross build) Windows CUDA build options: `-DCMAKE_TOOLCHAIN_FILE=../toolchain-mingw-cuda-win64.cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;120" -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON` (without the last six a cross build has no SIMD in the CPU backend)

"Built" means compiled and linked. The v0.8.1 user logs now verify GPU execution and completed requests on the three PCs. This does not establish comprehensive model answer quality or make the historical performance estimates measured limits.
