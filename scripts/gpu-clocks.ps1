# Lock this PC's NVIDIA GPU clocks at full speed for a LlamaGLM session, or reset them with -Reset.
# During generation each GPU is busy only about 10% of the time, so the driver keeps it in P8/P5 with
# 405-810 MHz memory clocks (2026-10-25 logs), which makes its layers several times slower.
# Needs an administrator window (an SSH session of an administrator account is one). The lock lasts until
# -Reset or a reboot; the launchers' -LockClocks switch runs this and resets when they exit.
param (
    [switch]$Reset,
    [string]$OutFile = ""
)

function Write-Out([string]$Text) {
    Write-Host $Text
    if ($OutFile) {
        Add-Content -Path $OutFile -Value $Text -Encoding UTF8
    }
}

function Invoke-Smi([string[]]$SmiArgs) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & nvidia-smi @SmiArgs 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    foreach ($l in $out) {
        if ("$l".Trim()) {
            Write-Out ("  nvidia-smi: {0}" -f "$l".Trim())
        }
    }
    return $code
}

function Get-MaxClock([string]$Kind) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $vals = @(& nvidia-smi "--query-supported-clocks=$Kind" --format=csv,noheader,nounits 2>$null |
        ForEach-Object { "$_".Trim() } | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ })
    $ErrorActionPreference = $prev
    if ($vals.Count -eq 0) {
        return 0
    }
    return ($vals | Measure-Object -Maximum).Maximum
}

if (!(Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Out "[WARN] nvidia-smi not found; GPU clocks unchanged."
    exit 1
}
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (!([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Out "[WARN] locking or resetting GPU clocks needs an administrator window; GPU clocks unchanged."
    exit 2
}

if ($Reset) {
    $c1 = Invoke-Smi @("--reset-memory-clocks")
    $c2 = Invoke-Smi @("--reset-gpu-clocks")
    Write-Out "[GPU CLOCKS] reset to the driver's defaults"
    exit $(if ($c1 -eq 0 -and $c2 -eq 0) { 0 } else { 1 })
}

$mem = Get-MaxClock "mem"
$gr  = Get-MaxClock "graphics"
if ($mem -le 0 -or $gr -le 0) {
    Write-Out "[WARN] this GPU does not report its supported clocks; GPU clocks unchanged."
    exit 1
}
# memory at its highest clock; graphics may still boost from 3/4 of its highest clock
$grMin = [int]($gr * 3 / 4)
$cm = Invoke-Smi @("--lock-memory-clocks=$mem,$mem")
$cg = Invoke-Smi @("--lock-gpu-clocks=$grMin,$gr")
[void](Invoke-Smi @("--query-gpu=name,pstate,clocks.sm,clocks.mem", "--format=csv,noheader"))
Write-Out ("[GPU CLOCKS] memory {0} MHz: {1}; graphics {2}-{3} MHz: {4}. Undo: .\gpu-clocks.ps1 -Reset (as administrator) or reboot." -f
    $mem, $(if ($cm -eq 0) { "locked" } else { "not supported" }), $grMin, $gr, $(if ($cg -eq 0) { "locked" } else { "not supported" }))
exit $(if ($cm -eq 0 -or $cg -eq 0) { 0 } else { 1 })
