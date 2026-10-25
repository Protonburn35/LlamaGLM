# Stage C worker. Same as: .\start-worker.ps1 -Stage C [-SkipSelfTest] [-Diagnostic] [-Restarts N] [-BinDir DIR] [-Timing] [-LocalCopy] [-NoCudaGraphs] [-LockClocks] [-ChunkedLoad]
param ([switch]$SkipSelfTest, [switch]$Diagnostic, [int]$Restarts = 3, [string]$BinDir = $PSScriptRoot, [switch]$Timing, [switch]$LocalCopy, [switch]$NoCudaGraphs, [switch]$LockClocks, [switch]$ChunkedLoad)
& (Join-Path $PSScriptRoot "start-worker.ps1") -Stage C -SkipSelfTest:$SkipSelfTest -Diagnostic:$Diagnostic -Restarts $Restarts -BinDir $BinDir -Timing:$Timing -LocalCopy:$LocalCopy -NoCudaGraphs:$NoCudaGraphs -LockClocks:$LockClocks -ChunkedLoad:$ChunkedLoad
exit $LASTEXITCODE
