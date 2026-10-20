# Build matrix

| Platform | Toolchain | Target | Execution evidence |
| --- | --- | --- | --- |
| Cloud Linux x86_64 | GCC 14.2, CMake 3.31.6, Ninja 1.11.1.4, AVX2 | Planner, ownership/GGUF tests, synthetic GLM fixture | Local pass; no GPU |
| Windows Server 2022 x64 CPU | MSVC 19.44.35229.0, static llama libraries, Ninja | Planner, real llama-server/llama-cli, ownership/GGUF/GLM tests | PASS at acae9b6aa; see BUILD_LOG |
| Windows Server 2022 x64 CUDA SM89 | Verified NVIDIA 12.8.1 components, pinned CCCL v3.4.3 commit | Same targets plus actual CUDA backend, 89-real, FP16/BF16 attention cache variants | Compile/link passed at acae9b6aa; DLL load fix/new source pending |
| Windows Server 2022 x64 CUDA SM120 | Same pinned toolchain | Same targets, 120a-real, FP16/BF16 attention cache variants | Compile/link passed at acae9b6aa; DLL load fix/new source pending |
| Physical Windows 11 Ada | RTX 4090/4060 | Real kernels and assigned target stages | NOT_RUN |
| Physical Windows 11 Blackwell | RTX 5070 Ti | Real kernels and terminal stage | NOT_RUN |

Hosted Windows runners have no NVIDIA inference qualification. A CPU fixture run in a CUDA-linked build is still CPU evidence. The workflow uses contents-read permission, two compile jobs per runner, three-day artifact retention and exact source-SHA artifact names. Current artifacts are development tools/monolithic upstream binaries, not a packaged three-PC application.

CUDA architectures now compile in separate ordinary hosted jobs to reduce critical-path time. The two requested attention cache precisions are compiled explicitly; quantized weight codecs remain included. Each job has a checksum-verified pinned ccache 4.13.6 binary and a fixed-key 2G compiler cache. The combined-architecture job compiled/linked successfully, then failed to load test-gguf because of a direct NVIDIA-driver DLL dependency. MSVC driver delay loading and explicit verified runtime paths address that setup defect; verification of the correction is pending.

New planner/expert-span changes after acae9b6aa require a subsequent Windows run; do not attribute the older job's pass to an unbuilt SHA. CUDA runtime redistribution, role launchers and a source/build release archive remain future distribution work.
