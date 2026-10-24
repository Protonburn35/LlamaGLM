# Collect LlamaGLM logs and a system snapshot into one zip (run on any of the 3 PCs).
# On Stage A it also copies the worker logs if the workers share their logs folder read-only:
#   on B and C, Admin PowerShell, once:
#   New-SmbShare -Name LlamaGLM-Logs -Path C:\LlamaGLM\logs -ReadAccess "$env:USERNAME"
# The share is only reachable on the LAN; do not forward port 445 on the router.
param (
    [string]$ConfigPath = (Join-Path $PSScriptRoot "cluster_config.json"),
    [switch]$NoWorkers
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "llamaglm-common.ps1")
$cfg = Get-LlamaGlmConfig $ConfigPath

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$work = Join-Path $env:TEMP "llamaglm-logs-$stamp"
New-Item -ItemType Directory -Force -Path $work | Out-Null

$logs = Join-Path $PSScriptRoot "logs"
if (Test-Path $logs) {
    Copy-Item -Path (Join-Path $logs "*.log") -Destination $work -ErrorAction SilentlyContinue
    Copy-Item -Path (Join-Path $logs "*.csv") -Destination $work -ErrorAction SilentlyContinue
}

$info = Join-Path $work ("system-{0}.txt" -f $env:COMPUTERNAME)
"LlamaGLM $($cfg.version) on $env:COMPUTERNAME at $(Get-Date -Format o)" | Out-File $info -Encoding utf8
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, TotalVisibleMemorySize, FreePhysicalMemory, TotalVirtualMemorySize, FreeVirtualMemory | Format-List | Out-File $info -Append -Encoding utf8
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage | Format-List | Out-File $info -Append -Encoding utf8
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi 2>&1 | Out-File $info -Append -Encoding utf8
}
Get-NetIPAddress -AddressFamily IPv4 | Select-Object IPAddress, InterfaceAlias | Format-Table | Out-File $info -Append -Encoding utf8
# recent display driver resets (TDR) and application crashes
Get-WinEvent -FilterHashtable @{ LogName = "System"; StartTime = (Get-Date).AddDays(-2) } -ErrorAction SilentlyContinue |
    Where-Object { $_.ProviderName -match "nvlddmkm|Display|WHEA" -or $_.Id -in 4101, 41 } |
    Select-Object -First 50 TimeCreated, ProviderName, Id, Message | Format-List | Out-File $info -Append -Encoding utf8
Get-WinEvent -FilterHashtable @{ LogName = "Application"; StartTime = (Get-Date).AddDays(-2); Id = 1000 } -ErrorAction SilentlyContinue |
    Where-Object { $_.Message -match "llama|ggml" } |
    Select-Object -First 20 TimeCreated, Message | Format-List | Out-File $info -Append -Encoding utf8

if (!$NoWorkers) {
    foreach ($w in @($cfg.stage_b, $cfg.stage_c)) {
        $share = "\\{0}\LlamaGLM-Logs" -f $w.host
        if (Test-Path $share) {
            $dst = Join-Path $work ("worker-" + $w.host)
            New-Item -ItemType Directory -Force -Path $dst | Out-Null
            Get-ChildItem -Path $share -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 10 | Copy-Item -Destination $dst
            Get-ChildItem -Path $share -Filter "*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | Copy-Item -Destination $dst
            Write-Host "copied worker logs from $share"
        } else {
            Write-Host "no log share at $share (optional, see the top of this script)" -ForegroundColor DarkGray
        }
    }
}

$zip = Join-Path $PSScriptRoot ("llamaglm-logs-{0}-{1}.zip" -f $env:COMPUTERNAME, $stamp)
Compress-Archive -Path (Join-Path $work "*") -DestinationPath $zip -Force
Remove-Item -Recurse -Force $work
Write-Host "Wrote $zip" -ForegroundColor Green
Write-Host "Review this archive before sharing: server logs and system details may contain private data." -ForegroundColor DarkGray
