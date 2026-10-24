# LlamaGLM RPC worker (Stage B laptop or Stage C desktop).
# Exposes the local GPU (CUDA0) and the local RAM (CPU) to the coordinator as two RPC devices.
param (
    [Parameter(Mandatory = $true)][ValidateSet("B", "C")][string]$Stage,
    [switch]$SkipSelfTest,
    [switch]$Diagnostic,
    [int]$Restarts = 3,
    # report the compute time of each device in the [stats] lines (GGML_RPC_SERVER_PROFILE)
    [switch]$Timing,
    # copy between this PC's RAM and GPU locally instead of through Stage A (GGML_RPC_SERVER_LOCAL_COPY)
    [switch]$LocalCopy,
    # folder with ggml-rpc-server.exe, e.g. the official llama.cpp b11780 CUDA build
    [string]$BinDir = $PSScriptRoot,
    [string]$ConfigPath = (Join-Path $PSScriptRoot "cluster_config.json")
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "llamaglm-common.ps1")

$cfg = Get-LlamaGlmConfig $ConfigPath
$w = if ($Stage -eq "B") { $cfg.stage_b } else { $cfg.stage_c }
$log = New-LlamaGlmLogPath ("stage-" + $Stage.ToLower())

Write-LogLine $log ("LlamaGLM {0} - Stage {1} worker ({2}) on {3}:{4}" -f $cfg.version, $Stage, $w.description, $w.host, $w.port) "Cyan"
Write-LogLine $log "Log file: $log" "DarkGray"

Test-RequiredFiles @("ggml-rpc-server.exe", "ggml-rpc.dll", "ggml-cuda.dll", "ggml-base.dll", "ggml.dll") $BinDir
$ownBuild = (Resolve-Path $BinDir).Path -eq (Resolve-Path $PSScriptRoot).Path
Clear-UnsafeEnv $log
Test-CpuSupport $log

# 1. the address in the config must belong to this PC, otherwise binding fails (DHCP may have changed it)
$ips = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | ForEach-Object { $_.IPAddress })
if ($ips -notcontains $w.host) {
    Write-LogLine $log ("[FAIL] {0} is not an address of this PC (found: {1}). Fix stage_{2}.host in cluster_config.json on ALL PCs, or reserve the address in the router." -f $w.host, ($ips -join ", "), $Stage.ToLower()) "Red"
    exit 1
}

# 2. GPU, driver and memory
if (!(Write-GpuInfo $log)) {
    Write-LogLine $log "[FAIL] No working NVIDIA driver found. Install a current GeForce driver (R570 or newer for CUDA 12.8)." "Red"
    exit 1
}

# 3. local CUDA kernel self-test with tiny tensors (catches broken kernels before any model data is sent)
# the self-test exercises the LlamaGLM CUDA build, so it is skipped for -BinDir
if (!$SkipSelfTest -and $ownBuild) {
    if (!(Invoke-CudaSelfTest $log)) {
        exit 1
    }
}

if ($Diagnostic) {
    Set-DiagnosticEnv $log
}

# 4. weight cache: later loads send only hashes and the worker reads weights from this folder
if (!(Test-Path $w.cache_dir)) {
    New-Item -ItemType Directory -Force -Path $w.cache_dir | Out-Null
}
$env:LLAMA_CACHE = $w.cache_dir
$drive = Get-PSDrive -Name ($w.cache_dir.Substring(0, 1))
Write-LogLine $log ("Weight cache: {0} ({1:N1} GiB free on {2}:, needs up to {3} GiB)" -f $w.cache_dir, ($drive.Free / 1GB), $drive.Name, $w.cache_need_gib) "Gray"
if ($drive.Free / 1GB -lt $w.cache_need_gib) {
    Write-LogLine $log "[WARN] Not enough disk space for the weight cache. Loading still works, but every load re-sends the weights over the LAN." "Yellow"
}

# 5. refuse allocations that would leave less than this free on the GPU / in RAM (GGML_RPC_SERVER_RESERVE_MIB)
$env:GGML_RPC_SERVER_RESERVE_MIB = $w.reserve_mib

# 6. optional experiments, off unless a switch or "options" in cluster_config.json turns them on
$opt = $cfg.options
$useTiming    = $Timing -or ($opt -and $opt.worker_timing)
$useLocalCopy = $LocalCopy -or ($opt -and $opt.worker_local_copy)
Set-OptionalEnv $log "GGML_RPC_SERVER_PROFILE" $useTiming
Set-OptionalEnv $log "GGML_RPC_SERVER_LOCAL_COPY" $useLocalCopy
Write-LogLine $log ("[OPTIONS] timing {0}, local copy {1}" -f $(if ($useTiming) { "on" } else { "off" }), $(if ($useLocalCopy) { "on" } else { "off" })) "Cyan"
$threads = $w.threads
if ($threads -le 0) {
    $threads = [int]((Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum)
}
Write-LogLine $log ("Devices: {0} (RPC device order seen by Stage A), CPU threads: {1}, reserve MiB (GPU,RAM): {2}" -f $w.devices, $threads, $w.reserve_mib) "Gray"
Write-LogLine $log ("Firewall: allow inbound TCP {0} from {1} only (Admin PowerShell, once):" -f $w.port, $cfg.stage_a.host) "Yellow"
Write-LogLine $log ("  New-NetFirewallRule -DisplayName 'LlamaGLM-RPC-{0}' -Direction Inbound -Protocol TCP -LocalPort {0} -RemoteAddress {1} -Action Allow" -f $w.port, $cfg.stage_a.host) "DarkGray"
Write-LogLine $log "[stats] lines below show what this worker received and computed. Keep this window open; Ctrl+C stops the worker." "Green"

$argList = @("-H", $w.host, "-p", "$($w.port)", "-d", $w.devices, "-t", "$threads", "-c")
# a crashed worker (for example a CUDA error) is restarted as a new process with a fresh CUDA context,
# so Stage A can simply be started again; the count is limited so a broken GPU does not loop forever
for ($run = 0; $run -le $Restarts; $run++) {
    $code = Invoke-Logged (Join-Path $BinDir "ggml-rpc-server.exe") $argList $log
    Write-LogLine $log "ggml-rpc-server exited with code $code" "Yellow"
    if ($code -eq 0 -or $run -eq $Restarts) {
        break
    }
    Write-LogLine $log ("Restarting in 15 s ({0} of {1}). Stage A must be restarted too. Send {2} if this repeats." -f ($run + 1), $Restarts, $log) "Yellow"
    Start-Sleep -Seconds 15
}
exit $code
