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
