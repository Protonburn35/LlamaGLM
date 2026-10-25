# Installing LlamaGLM on Windows 11

The step-by-step run is in `FIRST-RUN.md`. This page lists what each PC needs.

## All three PCs
- Windows 11 x64, NVIDIA driver R570 or newer. (The optional `-BinDir` reference build of llama.cpp b11780 with CUDA 13.4 needs R580 or newer.)
- `C:\LlamaGLM` = contents of `LlamaGLM-v0.8.3-win-cuda-cu12.8-sm89-sm120-x64.zip`. Same version on all PCs; remove older versions first. `MANIFEST.txt` lists the SHA-256 of every file (`Get-FileHash <file> -Algorithm SHA256`).
- NVIDIA Control Panel: "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback".
- PowerShell may need `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and `Get-ChildItem C:\LlamaGLM\*.ps1 | Unblock-File`.
- `cluster_config.json` must be identical on all PCs (addresses, ports).

## Stage A (192.168.50.206, RTX 4090)
- Model: `C:\GLM-5.3-Flash\UD-IQ3_XXS\GLM-5.3-Flash-UD-IQ3_XXS-0000{1..4}-of-00004.gguf`, 112.10 GiB total (9,429,984 / 49,261,511,584 / 49,075,832,480 / 22,020,797,792 bytes). Download: `hf download unsloth/GLM-5.3-Flash-GGUF --include "UD-IQ3_XXS/*" --local-dir "C:\GLM-5.3-Flash"`. An SSD is strongly recommended: about 35 GiB of the model is read through mmap during inference.
- No inbound firewall rule is needed (the API listens on 127.0.0.1 only).

## Stages B (192.168.50.129) and C (192.168.50.143)
- No model files.
- Free disk for the worker weight cache `C:\LlamaGLM-Cache`: about 26 GiB on B, 32 GiB on C.
- Free RAM: about 24 GiB when the model loads (close other programs).
- Firewall (Admin PowerShell, once), only Stage A may connect:
  ```powershell
  # B
  New-NetFirewallRule -DisplayName LlamaGLM-RPC-50052 -Direction Inbound -Protocol TCP -LocalPort 50052 -RemoteAddress 192.168.50.206 -Action Allow
  # C
  New-NetFirewallRule -DisplayName LlamaGLM-RPC-50053 -Direction Inbound -Protocol TCP -LocalPort 50053 -RemoteAddress 192.168.50.206 -Action Allow
  ```
- Fixed IP addresses (router DHCP reservation) avoid editing `cluster_config.json`.

## Package contents
`llama-server.exe`, `ggml-rpc-server.exe`, `llama-fit-params.exe` (Gate 2), `llama-rpc-test.exe` (Gate 1), `test-backend-ops.exe` (CUDA self-test), `llama-cli.exe`, `llama-bench.exe`, `llama-gguf-split.exe`, the ggml/llama DLLs (`ggml-cuda.dll` with sm_89 + sm_120a code), CUDA 12.8 runtime DLLs, the Microsoft Visual C++ runtime DLLs, the launchers (`start-stage-a.ps1`, `start-stage-b.ps1`, `start-stage-c.ps1`, `start-worker.ps1`, `smoke-test.ps1`, `collect-logs.ps1`, `perf-log.ps1`, `llamaglm-common.ps1`), `cluster_config.json`, `MANIFEST.txt` (source commit and file hashes), docs.

Logs go to `C:\LlamaGLM\logs` on each PC.

## Source checkout layout

The GitHub checkout contains the engine at its root, with launchers and `cluster_config.json` under `scripts/`. The VM export's nested `LlamaGLM-master/` directory has been flattened. When assembling a binary folder, copy `scripts/*.ps1` and `scripts/cluster_config.json` beside the executables. `scripts/package-release.sh` uses `build-win-cuda/` by default, or the directory selected by `LLAMAGLM_BUILD_DIR`.

The Windows package is built by `.github/workflows/llamaglm-package-windows.yml` (run it by hand in GitHub Actions; the package is the run's artifact) with MSVC and nvcc. `.github/workflows/llamaglm-rpc-windows.yml` only produces development executables for build verification. The earlier v0.8.0 archive was a clang cross-build on the VM; it does not include the follow-up source fixes listed in `docs/REVIEW-v0.8.0.md`.
