param([string]$BackendOps = "")
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "../scripts/llamaglm-common.ps1")
$testDir = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $testDir | Out-Null
$log = Join-Path $testDir "self-test.log"
try {
    if (Invoke-CudaSelfTest -LogPath $log -TestExecutable (Join-Path $testDir "absent.exe")) {
        throw "Missing test executable passed"
    }
    if ($BackendOps) {
        if (Invoke-CudaSelfTest -LogPath $log -TestExecutable $BackendOps) {
            throw "CPU-only backend executable passed the CUDA gate"
        }
    }
    $fakeExe = Join-Path $testDir "mock.exe"
    New-Item -ItemType File -Path $fakeExe | Out-Null
    function Start-Process {
        param($FilePath, $ArgumentList, [switch]$NoNewWindow, [switch]$PassThru,
              $RedirectStandardOutput, $RedirectStandardError)
        [System.IO.File]::WriteAllText($RedirectStandardOutput, $script:testOutput)
        [System.IO.File]::WriteAllText($RedirectStandardError, "")
        $process = [pscustomobject]@{ Handle = 1; ExitCode = 0 }
        $process | Add-Member ScriptMethod WaitForExit { if ($args.Count -gt 0) { return $true } }
        $process | Add-Member ScriptMethod Kill { }
        return $process
    }
    $cases = @(
        @{ Text = "Testing 1 devices`nBackend 1/1: CPU`n  Skipping`n1/1 backends passed`nOK"; Pass = $false },
        @{ Text = "  0/0 tests passed`n  Backend CUDA0: OK"; Pass = $false },
        @{ Text = "  8/9 tests passed`n  Backend CUDA0: FAIL"; Pass = $false },
        @{ Text = "  9/9 tests passed`n  Backend CPU: OK"; Pass = $false },
        @{ Text = "  9/9 tests passed`n  Backend CUDA0: $([char]27)[1;32mOK$([char]27)[0m"; Pass = $true }
    )
    foreach ($case in $cases) {
        $script:testOutput = $case.Text
        $actual = Invoke-CudaSelfTest -LogPath $log -TestExecutable $fakeExe
        if ($actual -ne $case.Pass) { throw "Unexpected CUDA gate result for: $($case.Text)" }
    }
    Write-Host "PASS: CUDA launcher refuses absent, skipped, empty and failed tests; accepts executed CUDA tests"
} finally {
    Remove-Item -LiteralPath $testDir -Recurse -Force
}

# every plan must place each slot exactly once and be selectable; every experiment must default to off
$cfg = Get-LlamaGlmConfig (Join-Path $PSScriptRoot "../scripts/cluster_config.json")
$launcher = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot "../scripts/start-stage-a.ps1"), [ref]$null, [ref]$null)
$planParam = $launcher.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq "Plan" }
$planNames = @($planParam.Attributes | Where-Object { $_.TypeName.Name -eq "ValidateSet" } | ForEach-Object { $_.PositionalArguments.Value })
foreach ($plan in $cfg.profiles.PSObject.Properties) {
    $p = $plan.Value
    if ((($p.ts | Measure-Object -Sum).Sum) -ne $p.ngl) { throw "plan $($plan.Name): sum(ts) != ngl" }
    if ($p.devices.Count -ne $p.ts.Count -or $p.fit_margin_mib.Count -ne $p.ts.Count) { throw "plan $($plan.Name): devices, ts and fit_margin_mib differ in length" }
    if ($planNames -notcontains $plan.Name) { throw "plan $($plan.Name) is not in the -Plan list of start-stage-a.ps1" }
    # with --n-cpu-moe every layer is offloaded and the CPU-expert layers must be on the 4090 (the first device)
    if ($p.n_cpu_moe -gt 0 -and ($p.ngl -ne 47 -or $p.ts[0] -lt $p.n_cpu_moe)) { throw "plan $($plan.Name): n_cpu_moe needs ngl 47 and at least n_cpu_moe layers on the first device" }
}
foreach ($name in @("mtp", "moe_prefetch", "timing", "op_offload", "fused_hc", "cuda_graphs_off", "lock_clocks", "worker_timing", "worker_local_copy", "worker_cuda_graphs_off", "worker_lock_clocks", "worker_chunked_load")) {
    if ($cfg.options.$name -ne $false) { throw "option $name must default to false" }
}
if ($cfg.options.moe_cache_mib -ne 0) { throw "option moe_cache_mib must default to 0" }
Write-Host "PASS: $(@($cfg.profiles.PSObject.Properties).Count) plans are consistent and selectable; all experiments default to off"
