# Build matrix

| Platform | Toolchain | Target | Execution evidence |
| --- | --- | --- | --- |
| Cloud Linux x86_64 | GCC 14.2, CMake 3.31.6, Ninja 1.11.1.4, AVX2 | Planner, native schema, ownership, authenticated transport/identity/bridge, three-process GLM and complete streamed-model fixtures | Eleven CTest plus six GLM tests pass; minimal streaming-disabled library build passes; no GPU |
| Windows Server 2022 x64 CPU | MSVC 19.44.35229.0, static llama libraries, Ninja | Planner, real llama-server/llama-cli, ownership/GGUF/GLM tests | PASS at 81f4a71cb, run 39546390989; ten CTest/six GLM/native real-header checks |
| Windows Server 2022 x64 CUDA SM89 | Verified NVIDIA 12.8.1 components, pinned CCCL v3.4.3 commit | Same targets plus actual CUDA backend, 89-real, FP16/BF16 attention cache variants | Compile/link and hosted CPU fixtures PASS at 81f4a71cb; GPU NOT_RUN |
| Windows Server 2022 x64 CUDA SM120 | Same pinned toolchain | Same targets, 120a-real, FP16/BF16 attention cache variants | Compile/link and hosted CPU fixtures PASS at 81f4a71cb; GPU NOT_RUN |
| Physical Windows 11 Ada | RTX 4090/4060 | Real kernels and assigned target stages | NOT_RUN |
| Physical Windows 11 Blackwell | RTX 5070 Ti | Real kernels and terminal stage | NOT_RUN |

Hosted Windows runners have no NVIDIA inference qualification. A CPU fixture run in a CUDA-linked build is still CPU evidence. The workflow uses contents-read permission, two compile jobs per runner, three-day artifact retention and exact source-SHA artifact names. Current artifacts are development tools/monolithic upstream binaries, not a packaged three-PC application.

CUDA architectures now compile in separate ordinary hosted jobs to reduce critical-path time. The two requested attention cache precisions are compiled explicitly; quantized weight codecs remain included. Each job has a checksum-verified pinned ccache 4.13.6 binary and a fixed-key 2G compiler cache. The combined-architecture job compiled/linked successfully, then failed to load test-gguf because of a direct NVIDIA-driver DLL dependency. MSVC driver delay loading and explicit verified runtime paths fixed that setup defect; native launch and hosted CPU fixtures have passed both architectures since 1410c4041. VMM remains enabled.

The best Windows-tested source is 81f4a71cb, with exact jobs/artifacts recorded in BUILD_LOG. CUDA runtime redistribution, role launchers and a source/build release archive remain future distribution work.
