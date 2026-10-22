# Test report

Imported VM report for the original v0.8.0 snapshot. Independently repeated checks of the GitHub import are listed in [REVIEW-v0.8.0.md](REVIEW-v0.8.0.md).

**Environment:** GCP VM, Linux 7.0.0-1014-gcp, Ubuntu 24.04, 8 vCPU, 31 GiB RAM, **no GPU**, gcc 13.3, cmake 3.28, pwsh 7.6.6, clang 20.1.2 (for PTX checks only).
**Nothing here ran on real Windows or on a GPU** (T21 runs the Windows binaries under Wine, CPU only). No test below shows that CUDA kernels work on the RTX 4090/4060/5070 Ti or that the real model generates text.

Engine under test: `LlamaGLM-master` at `502ff59` (repaired) and, as baseline, `5453314` (Gemini's last commit). Both built for Linux CPU in the scratch area:

```bash
git archive <commit> | tar -x -C SRC
cmake -B SRC/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
ninja -C SRC/build ggml-rpc-server llama-server llama-completion test-llama-archs llama-fit-params llama-rpc-test test-backend-ops
```
Result: both build without errors.

Test models: upstream `test-llama-archs -a glm5-next -o DIR` (random weights). A scratch-only patch (env `GLM5_TEST_LAYERS=8`, not committed) makes an 8-layer variant: 9 slots, 396 tensors, 19.7 MiB. Cluster stand-in: two `ggml-rpc-server -H 127.0.0.1 -p 50152|50153 -d CPU,CPU -t 2` (2 devices each, like a worker's GPU + RAM), plus a third (`-p 50154 -d CPU`) for the 4090 slot where needed.

| ID | Test | Result |
|---|---|---|
| T1 | Baseline 5453314: `llama-completion --rpc ... --device RPC0,RPC1,RPC2,RPC3 -ngl 4 -ts 1,1,1,1` | Generates, but log shows **`load_mode = none`**: mmap disabled by the RPC caps change (root cause of RAM overcommit). |
| T2 | Repaired: same with `-ngl 6 -ts 1,2,1,2` | Generates; **`load_mode = mmap`**; placement lines `layers 3-3 -> RPC0`, `4-5 -> RPC1`, `6-6 -> RPC2`, `7 + output -> RPC3`: exactly the `-ts` counts. |
| T3 | Admission: `-fitt 1024,40000,1024,1024` | `RPC1 ... DOES NOT FIT`, "not loading the model", exit 1, no buffers allocated. |
| T4 | Host rule: `-ngl 0 -fitt 32050` with mmap / with `-lm none` | mmap: host model not counted -> ok; `-lm none`: host model counted -> DOES NOT FIT, exit 1. |
| T5 | Worker reserve: server B with `GGML_RPC_SERVER_RESERVE_MIB=32080`, client `--fit-check off` | Worker logs `refusing 20.11 MiB, only 31.34 GiB free and 31.33 GiB must stay free`, keeps running; client: `unable to allocate RPC1[127.0.0.1:50152] buffer`, exit 1. |
| T6 | Gate 1: `llama-rpc-test --devices 2 --expect EP/0=25:40 ... EP_B EP_C` | 4 devices x (MUL_MAT, MUL_MAT_ID) x (iq2_s, iq3_s, iq4_xs, q6_K, q8_0, f16) x (n=1, 8): all ok, exit 0. |
| T7 | Gate 1 negative: `--expect 127.0.0.1:50152/0=6:9` (device 0 must look like an 8 GB GPU) | `[FAIL] total memory 31.34 GiB is outside the expected 6.0-9.0 GiB`, exit 1 (this is what a worker that fell back to CPU looks like). |
| T8 | **Real model metadata**: sparse copies of the 4 real shards (`cp` header + `truncate -s <exact size>`, 105 MB on disk); `llama-fit-params -m <shard 1> --rpc A',B,C --device RPC4,RPC1,RPC0,RPC3,RPC2 -ngl 31 -ts 7,8,2,8,6 -c 8192 -ub 512 -np 1 --fit off -fitt ... --fit-print on` | Loader accepts the real GLM-5.3-Flash GGUF. Model bytes per device 18200 / 20801 / 5200 / 20801 / 11130 MiB, host 35977 MiB mmapped: identical to the plan computed from the tensor table. KV/state 18-53 MiB per device at 8K. 1.1 s, 185 MB RSS. Same for `laptop-light`. |
| T9 | `llama-server` with the launcher's flags (`--rpc`, reordered `--device RPC1,RPC0,RPC3,RPC2`, `-np 1 -cram 0 --fit off -fitt ... -lv 4 --metrics --log-file`) | `/health` ok after 2 s; `/completion` 12 tokens; chat 12 tokens. Worker `[stats]` at disconnect: 741 graphs per device on B. |
| T10 | `smoke-test.ps1` (pwsh 7) against T9 with `-MinRpcBufferMiB 0` | Gate 3 PASS (mmap, 4 RPC buffers), Gate 4 PASS (16 raw tokens, 237 chat tokens), Gate 5 PASS (token counter 0 -> 735). With `-lv 3` Gate 3 correctly FAILED (libllama INFO lines need `-lv 4`; fixed in the launcher). |
| T11 | `start-stage-a.ps1 -PlanOnly -SkipSelfTest` (pwsh 7; Windows-only cmdlets stubbed; `.exe` names linked to the Linux binaries; config pointed at T8's sparse model and the local servers) | Gate 0 shard size check, Gate 1, Gate 2 PASS, exit 0. Negative cases: margin 40000 -> Gate 2 FAIL exit 1; `ts` sum != `ngl` -> FAIL before any projection. Same with `-BinDir <other folder>`. |
| T12 | `Invoke-Logged` with a native command that writes stdout + stderr and exits 3, `$ErrorActionPreference = "Stop"` | Both lines echoed and logged with timestamps, exit code 3 returned. |
| T13 | PowerShell parse check (`[Parser]::ParseFile`) of all `scripts/*.ps1` | 0 errors. A `"$BinDir:"` interpolation bug was caught here and fixed. |
| T14 | PTX check of the clang device flags: `topk-moe.cu`, `softmax.cu` and a 3-line kernel compiled for sm_89 with the toolchain flags | Gemini's `-ffast-math`: `x == -INFINITY` compiled to constant 0; topk-moe 206 / softmax 47 `-Wnan-infinity-disabled` warnings; PTX differs. With `-fno-finite-math-only`: 0 warnings, real compares. |
| T15 | Server log privacy at `-lv 4` | No prompt text in the server log. |
| T16 | CPU backend SIMD: `objdump -d ggml-cpu.dll`, count `vpmaddubsw/vfmadd*ps/vpdpbusd` | Gemini's shipped v0.7.0 DLL: **0** (and no `ymm` use at all: scalar/SSE2 only). MinGW cross build of `502ff59` with `-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON ...` (targets `ggml-cpu ggml-rpc-server llama-rpc-test`): builds, 1972. |
| T17 | `Invoke-CudaSelfTest` with a fake `test-backend-ops.exe` (pass / one FAIL line exit 1 / CUDA error exit 134) | returns True / False / False with the failing lines logged. Found and fixed before: `Start-Process -PassThru` leaves `ExitCode` empty unless the process handle is read early (a passing self-test would have been reported as failed), and case-insensitive `-match "FAIL"`. |
| T18 | Final scripts: T11 and T10 repeated after the last changes (`--no-repack`, CPU check, self-test fix) | Gates 0-2 PASS, Gates 3-5 PASS. |
| T19 | `smoke-test.ps1` answer check (2026-10-22) against a stub HTTP server: right answers / garbage raw text / right raw text but no "4" in the chat answer / garbage with `-NoAnswerCheck` | PASS, exit 0 / `[FAIL] raw completion does not mention Paris`, exit 1 / `[FAIL] chat answer to 2 + 2 contains no 4`, exit 1 / PASS with "tokens (answers not checked)". Before this, Gate 4 passed any tokens, so garbage output from a broken kernel would have counted as success. |
| T20 | **Windows CUDA build v0.8.0** (2026-10-22, `502ff59`): see `BUILD_LOG.md` | Builds, 0 errors, 0 `Wnan-infinity-disabled`. `ggml-cpu.dll` has AVX2/FMA code (2706 key instructions), `ggml-cuda.dll` has sm_89 + sm_120a images in all 141 fatbins, build commit = HEAD. |
| T21 | Windows binaries of T20 under **Wine 9.0** on the VM (CPU only; `ggml-cuda.dll` does not load because there is no NVIDIA driver): (a) `ggml-rpc-server.exe -d CPU,CPU` serving the Linux `llama-completion` client with the 8-layer GLM5-next model, `--device RPC0,RPC1 -ngl 99 -ts 1,1`, greedy; (b) `llama-rpc-test.exe --devices 2` against it; (c) `llama-server.exe` with the launcher's flags on the same model | (a) output **byte-identical** to the Linux CPU-only reference; worker `[stats]` lines printed; (b) `GATE1 PASSED`; (c) `common_fit_check` lines, `load_mode = mmap`, `/health` ok, 16 tokens identical to the reference. Shows the MinGW-built AVX2 CPU backend, the Windows RPC server/client and the server work. Says nothing about the CUDA kernels. Note: `llama-server.exe --version` prints but does not exit under Wine (`llama-cli.exe --version` does); not used by the launchers. |

## Not tested (needs the user's PCs)

- Any CUDA kernel on sm_89 / sm_120 (the per-GPU self-test in the launchers does this first).
- The Windows CUDA build of the repaired source on a real GPU (built 2026-10-22, T20; CUDA kernels not run).
- Windows PowerShell 5.1 (scripts avoid PS7-only syntax; `Invoke-Logged` relaxes `ErrorActionPreference` for native stderr).
- The real 112 GiB load, LAN transfer time, warm-up, token generation, speed, Stage A SSD paging.
- The previous `test-layer-split` and Gate 1 results in older versions of this file tested code that has been removed.
