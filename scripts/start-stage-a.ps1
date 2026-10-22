# LlamaGLM Stage A: coordinator + OpenAI-compatible API on the RTX 4090 PC.
# Runs the safe checks first and stops at the first failure:
#   Gate 0  files, driver, model shards
#   Gate 1  local CUDA self-test, then remote RPC device check (small tensors only)
#   Gate 2  memory projection for the real model on every device (no allocation)
# Only then it starts llama-server. Use -PlanOnly to stop after Gate 2.
param (
    [ValidateSet("balanced", "laptop-light")][string]$Plan = "balanced",
    [string]$ModelDir = "",
    [switch]$PlanOnly,
    [switch]$SkipSelfTest,
    [switch]$SkipRpcTest,
    [switch]$Diagnostic,
    [switch]$OpOffload,
    # folder with llama-server.exe / llama-fit-params.exe, e.g. the official llama.cpp b11780 CUDA build
    [string]$BinDir = $PSScriptRoot,
    [string]$ConfigPath = (Join-Path $PSScriptRoot "cluster_config.json")
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "llamaglm-common.ps1")

$cfg = Get-LlamaGlmConfig $ConfigPath
$p = $cfg.profiles.$Plan
$log = New-LlamaGlmLogPath "stage-a"
$nodeB = "{0}:{1}" -f $cfg.stage_b.host, $cfg.stage_b.port
$nodeC = "{0}:{1}" -f $cfg.stage_c.host, $cfg.stage_c.port

Write-LogLine $log ("LlamaGLM {0} - Stage A coordinator, plan '{1}': {2}" -f $cfg.version, $Plan, $p.note) "Cyan"
Write-LogLine $log "Log file: $log" "DarkGray"
foreach ($k in $p.layers.PSObject.Properties) {
    Write-LogLine $log ("  {0,-22} layers {1}" -f $k.Name, $k.Value) "Gray"
}

# ---- Gate 0: files, driver, model --------------------------------------------------------------
Test-RequiredFiles @("llama-rpc-test.exe", "ggml-rpc.dll", "ggml-cpu.dll", "ggml-base.dll", "ggml.dll")
Test-RequiredFiles @("llama-server.exe", "llama-fit-params.exe", "ggml-cuda.dll", "ggml-rpc.dll", "libllama.dll") $BinDir
$ownBuild = (Resolve-Path $BinDir).Path -eq (Resolve-Path $PSScriptRoot).Path
if (!$ownBuild) {
    Write-LogLine $log "[NOTE] -BinDir ${BinDir}: upstream binaries print the Gate 2 projection but cannot refuse a plan that does not fit; read the numbers yourself." "Yellow"
}
Clear-UnsafeEnv $log
Test-CpuSupport $log
if (!(Write-GpuInfo $log)) {
    Write-LogLine $log "[FAIL] No working NVIDIA driver found. Install a current GeForce driver (R570 or newer for CUDA 12.8)." "Red"
    exit 1
}
Write-LogLine $log "[NOTE] Set NVIDIA Control Panel > Manage 3D settings > CUDA - Sysmem Fallback Policy = 'Prefer No Sysmem Fallback' on all 3 PCs, so a full GPU fails fast instead of spilling into shared RAM." "Yellow"

if ([string]::IsNullOrEmpty($ModelDir)) {
    $ModelDir = $cfg.model.dir
}
$first = Join-Path $ModelDir $cfg.model.first_shard
$prefix = $cfg.model.first_shard -replace "-00001-of-[0-9]{5}\.gguf$", ""
$total = 0
$missing = @()
for ($i = 1; $i -le $cfg.model.shards; $i++) {
    $f = Join-Path $ModelDir ("{0}-{1:D5}-of-{2:D5}.gguf" -f $prefix, $i, $cfg.model.shards)
    if (Test-Path $f) {
        $total += (Get-Item $f).Length
    } else {
        $missing += $f
    }
}
if ($missing.Count -gt 0) {
    Write-LogLine $log ("[FAIL] Missing model shard(s): {0}" -f ($missing -join ", ")) "Red"
    Write-LogLine $log "  hf download unsloth/GLM-5.3-Flash-GGUF --include `"UD-IQ3_XXS/*`" --local-dir `"C:\GLM-5.3-Flash`"" "Gray"
    exit 1
}
Write-LogLine $log ("[OK] {0} shards, {1:N2} GiB (expected {2} GiB)" -f $cfg.model.shards, ($total / 1GB), $cfg.model.total_gib) "Green"
if ([math]::Abs($total / 1GB - $cfg.model.total_gib) -gt 0.5) {
    Write-LogLine $log "[FAIL] Model size does not match UD-IQ3_XXS. Re-download incomplete shards." "Red"
    exit 1
}

# ---- Gate 1: local CUDA kernels, then RPC workers ------------------------------------------------
if (!$SkipSelfTest -and $ownBuild) {
    if (!(Invoke-CudaSelfTest $log)) {
        exit 1
    }
}
foreach ($ep in @($cfg.stage_b, $cfg.stage_c)) {
    $ok = Test-NetConnection -ComputerName $ep.host -Port $ep.port -InformationLevel Quiet -WarningAction SilentlyContinue
    if (!$ok) {
        Write-LogLine $log ("[FAIL] {0}:{1} ({2}) is not reachable. Start its worker (start-stage-b.ps1 / start-stage-c.ps1) and check its firewall rule." -f $ep.host, $ep.port, $ep.description) "Red"
        exit 1
    }
}
if (!$SkipRpcTest) {
    $a = @("--devices", "2")
    foreach ($pair in @(@($nodeB, $cfg.stage_b), @($nodeC, $cfg.stage_c))) {
        for ($d = 0; $d -lt 2; $d++) {
            $a += @("--expect", ("{0}/{1}={2}" -f $pair[0], $d, $pair[1].expect_gib[$d]))
        }
    }
    $a += @($nodeB, $nodeC)
    Write-LogLine $log "[GATE 1] remote devices: count, memory size, quantized MUL_MAT / MUL_MAT_ID vs local CPU" "Cyan"
    $code = Invoke-Logged (Join-Path $PSScriptRoot "llama-rpc-test.exe") $a $log
    if ($code -ne 0) {
        Write-LogLine $log "[FAIL] Gate 1 failed. Check the worker windows; a worker without its GPU shows only one device or a RAM-sized device 0." "Red"
        exit 1
    }
    Write-LogLine $log "[PASS] Gate 1" "Green"
}

if ($Diagnostic) {
    Set-DiagnosticEnv $log
}

# ---- arguments shared by the projection and the server --------------------------------------------
if ((($p.ts | Measure-Object -Sum).Sum) -ne $p.ngl) {
    Write-LogLine $log ("[FAIL] plan '{0}': sum of ts ({1}) must equal ngl ({2}), otherwise layers shift between devices." -f $Plan, ($p.ts -join ","), $p.ngl) "Red"
    exit 1
}
$common = @(
    "-m", $first,
    "--rpc", "$nodeB,$nodeC",
    "--device", ($p.devices -join ","),
    "-ngl", "$($p.ngl)",
    "-ts", ($p.ts -join ","),
    "-c", "$($cfg.model.ctx)",
    "-ub", "$($cfg.model.ubatch)",
    "-np", "1",
    "--fit", "off",
    "-fitt", ($p.fit_margin_mib -join ","),
    # repacked weights (Q6_K, Q8_0, ~2.8 GiB here) would be copied out of the mmap into normal RAM
    "--no-repack",
    "-lv", "4"
)
# op offload copies whole CPU-resident expert tensors (up to 1.3 GiB each) into the 4090 compute buffer
# during prompt processing; keep it off for the first runs so VRAM use stays predictable
if (!$OpOffload) {
    $common += @("--no-op-offload")
}

# ---- Gate 2: projected memory of the real model on every device ----------------------------------
Write-LogLine $log "[GATE 2] projected weights + KV + compute per device vs free memory (nothing is allocated)" "Cyan"
$code = Invoke-Logged (Join-Path $BinDir "llama-fit-params.exe") ($common + @("--fit-print", "on")) $log
if ($code -ne 0) {
    Write-LogLine $log "[FAIL] Gate 2: the plan does not fit (see 'DOES NOT FIT' lines above). Close programs on that PC or move one layer from that device to the Stage A RAM range in cluster_config.json (keep sum(ts) = ngl)." "Red"
    exit 1
}
Write-LogLine $log "[PASS] Gate 2" "Green"
if ($PlanOnly) {
    exit 0
}

# ---- start the server -------------------------------------------------------------------------------
$serverLog = New-LlamaGlmLogPath "llama-server"
$server = $common + @(
    "--host", $cfg.stage_a.api_host,
    "--port", "$($cfg.stage_a.api_port)",
    "--alias", $cfg.model.alias,
    "-cram", "0",
    "--metrics",
    "--log-file", $serverLog,
    "--log-timestamps"
)
Write-LogLine $log ("Starting llama-server on http://{0}:{1}, server log: {2}" -f $cfg.stage_a.api_host, $cfg.stage_a.api_port, $serverLog) "Green"
Write-LogLine $log "First load sends about 56 GiB to the workers (about 8-10 min on 1 GbE); later loads use the worker caches. Then run .\smoke-test.ps1 in a second window." "Yellow"
Add-Content -Path $log -Value ("{0} >>> llama-server.exe {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"), ($server -join " ")) -Encoding UTF8
& (Join-Path $BinDir "llama-server.exe") @server
$code = $LASTEXITCODE
Write-LogLine $log "llama-server exited with code $code" "Yellow"
exit $code
