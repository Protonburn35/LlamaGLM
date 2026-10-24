# Log Windows performance counters once per second to logs\perf-<role>-<time>.csv; Ctrl+C stops.
# Start it in a second window after the server (A) or worker (B, C) is running, before sending a request.
# A watches llama-server (SSD reads, page faults), B and C watch ggml-rpc-server (CPU, paging).
# Counter names are English; on a Windows installed in another language typeperf reports them as not found.
param (
    [ValidateSet("A", "B", "C")][string]$Role = "A",
    [ValidateRange(1, 60)][int]$IntervalSec = 1
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "llamaglm-common.ps1")

$proc = if ($Role -eq "A") { "llama-server" } else { "ggml-rpc-server" }
if (!(Get-Process -Name $proc -ErrorAction SilentlyContinue)) {
    Write-Host "$proc is not running yet; its process counters would be missing. Start it first." -ForegroundColor Yellow
    exit 1
}
$out = (New-LlamaGlmLogPath ("perf-" + $Role.ToLower())) -replace "\.log$", ".csv"
$counters = @(
    "\Process($proc)\% Processor Time",
    "\Process($proc)\Page Faults/sec",
    "\Process($proc)\Working Set",
    "\Memory\Available MBytes",
    "\Memory\Page Reads/sec",
    "\Memory\Pages Input/sec",
    "\PhysicalDisk(_Total)\Disk Read Bytes/sec",
    "\PhysicalDisk(_Total)\Avg. Disk Read Queue Length",
    "\Processor(_Total)\% Processor Time"
)
Write-Host "Logging to $out every $IntervalSec s. Ctrl+C stops." -ForegroundColor Green
& typeperf @counters -si $IntervalSec -o $out -y
