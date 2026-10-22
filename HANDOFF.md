# LlamaGLM status

The active implementation on `master` is the repaired v0.8.0 upstream-RPC prototype exported from the VM (engine snapshot `502ff59`, launcher snapshot `6664b7d`). The engine now lives at the repository root; launchers are in `scripts/`. The original custom stage-runtime branch remains separate.

RPC intentionally uses the upstream unauthenticated protocol. The accepted prototype configuration uses 8K context and the documented RAM/mmap placement. See [FIRST-RUN.md](FIRST-RUN.md) and [LIMITATIONS.md](LIMITATIONS.md).

The import review fixed a CUDA self-test that could pass without executing CUDA tests, worker-reserve integer overflow and invalid settings, the packaging script's nested-repository assumption, and an overly broad Clang-only shim dependency. Launcher regression checks and actual local RPC tests passed on CPU. [Review and validation](docs/REVIEW-v0.8.0.md) distinguishes these checks from the original VM reports.

Native Windows CPU and SM89/SM120 compilation is defined in `.github/workflows/llamaglm-rpc-windows.yml`. GitHub-hosted runners exercise CPU fixtures; CUDA execution, full-model inference, and operation on the three PCs still need hardware validation.

CI follow-up for the next developer: the inherited upstream triggers launched dozens of unrelated workflows on the first `master` push. Cleanup is explicitly deferred. Read [docs/CI_SCOPE.md](docs/CI_SCOPE.md) before another source push; preserve useful AMD/Vulkan compatibility coverage in the eventual plan. Cancelled runs are not passing checks.
