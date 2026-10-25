# Shared helpers for the LlamaGLM launchers. Dot-source it: . "$PSScriptRoot\llamaglm-common.ps1"
# Works with Windows PowerShell 5.1 and PowerShell 7.

function Get-LlamaGlmConfig {
    param([string]$Path = (Join-Path $PSScriptRoot "cluster_config.json"))
    if (!(Test-Path $Path)) {
        throw "cluster_config.json not found at '$Path'. Run the launchers from C:\LlamaGLM."
    }
    return (Get-Content -Raw -Path $Path | ConvertFrom-Json)
}

function New-LlamaGlmLogPath {
    param([string]$Name)
    $dir = Join-Path $PSScriptRoot "logs"
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    return (Join-Path $dir ("{0}-{1}.log" -f $Name, (Get-Date -Format "yyyyMMdd-HHmmss")))
}

function Write-LogLine {
    param([string]$LogPath, [string]$Text, [string]$Color = "Gray")
    Write-Host $Text -ForegroundColor $Color
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"), $Text
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

# Run a native program, echo its output and append every line with a timestamp to $LogPath.
# Returns the exit code. stderr is merged; ErrorActionPreference is relaxed so that
# Windows PowerShell 5.1 does not turn the first stderr line into a terminating error.
function Invoke-Logged {
    param([string]$Exe, [string[]]$Arguments, [string]$LogPath)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $sw = New-Object System.IO.StreamWriter($LogPath, $true, (New-Object System.Text.UTF8Encoding($false)))
    $sw.AutoFlush = $true
    try {
        $sw.WriteLine(("{0} >>> {1} {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"), $Exe, ($Arguments -join " ")))
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $text = "$_"
            $sw.WriteLine(("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"), $text))
            Write-Host $text
        }
        $code = $LASTEXITCODE
        $sw.WriteLine(("{0} <<< exit code {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"), $code))
        return $code
    } finally {
        $sw.Close()
        $ErrorActionPreference = $prev
    }
}

function Test-RequiredFiles {
    param([string[]]$Files, [string]$Dir = $PSScriptRoot)
    $missing = @()
    foreach ($f in $Files) {
        if (!(Test-Path (Join-Path $Dir $f))) {
            $missing += $f
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing files in $Dir : $($missing -join ', '). Extract the whole release zip (and for -BinDir the matching cudart zip) there."
    }
}

# Log GPU, driver and memory state. Returns $false if nvidia-smi is missing or reports no GPU.
function Write-GpuInfo {
    param([string]$LogPath)
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (!$smi) {
        Write-LogLine $LogPath "[WARN] nvidia-smi not found; cannot check the NVIDIA driver." "Yellow"
        return $false
    }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $rows = & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader 2>&1
    $smiCode = $LASTEXITCODE
    $ErrorActionPreference = $prev
    foreach ($r in $rows) {
        Write-LogLine $LogPath "  GPU: $r" "Gray"
    }
    $os = Get-CimInstance Win32_OperatingSystem
    Write-LogLine $LogPath ("  RAM: {0:N1} GiB total, {1:N1} GiB free" -f ($os.TotalVisibleMemorySize / 1MB), ($os.FreePhysicalMemory / 1MB)) "Gray"
    return ($smiCode -eq 0)
}

# Small CUDA kernel self-test on the local GPU with the ops and weight types GLM-5.3-Flash uses.
# Tensors are tiny, so a bad kernel crashes this process instead of a full model load.
function Invoke-CudaSelfTest {
    param([string]$LogPath, [int]$TimeoutMin = 20,
          [string]$TestExecutable = (Join-Path $PSScriptRoot "test-backend-ops.exe"))
    $exe = $TestExecutable
    if (!(Test-Path $exe)) {
        Write-LogLine $LogPath "[FAIL] test-backend-ops.exe not in the package; CUDA self-test cannot run." "Red"
        return $false
    }
    $runs = @(
        @("-b", "CUDA0", "-o", "MUL_MAT,MUL_MAT_ID", "-p", "type_a=(iq2_s|iq3_s|iq4_xs|q6_K|q8_0)"),
        @("-b", "CUDA0", "-o", "FLASH_ATTN_EXT", "-p", "hsk=512"),
        @("-b", "CUDA0", "-o", "SOFT_MAX,ARGSORT,TOP_K,GATED_DELTA_NET,SSM_CONV,DSV4_HC_PRE,DSV4_HC_POST,DSV4_HC_COMB,LIGHTNING_INDEXER,GET_ROWS,SET_ROWS,RMS_NORM,ADD_ID")
    )
    foreach ($a in $runs) {
        Write-LogLine $LogPath ("[SELF-TEST] test-backend-ops {0}" -f ($a -join " ")) "Cyan"
        $out = [System.IO.Path]::GetTempFileName()
        $err = [System.IO.Path]::GetTempFileName()
        $p = Start-Process -FilePath $exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
        # ExitCode stays $null unless the process handle is read while the process is still running
        $null = $p.Handle
        if (!$p.WaitForExit($TimeoutMin * 60 * 1000)) {
            $p.Kill()
            Write-LogLine $LogPath "[FAIL] CUDA self-test timed out after $TimeoutMin min." "Red"
            return $false
        }
        $p.WaitForExit()
        $text = @(Get-Content $out) + @(Get-Content $err)
        Add-Content -Path $LogPath -Value $text -Encoding UTF8
        Remove-Item $out, $err -ErrorAction SilentlyContinue
        $text = @($text | ForEach-Object { $_ -replace '\x1b\[[0-9;]*m', '' })
        $failed = @($text | Where-Object { $_ -cmatch "\bFAIL\b" -or $_ -match "CUDA error|illegal memory|unspecified launch" })
        $cudaPassed = @($text | Where-Object { $_ -match '^\s*Backend CUDA0:\s*OK\s*$' }).Count -gt 0
        $ranTests = @($text | Where-Object { $_ -match '^\s*[1-9][0-9]*/[1-9][0-9]* tests passed\s*$' }).Count -gt 0
        $summary = @($text | Where-Object { $_ -match "tests passed|Backend .*: (OK|FAIL)" })
        foreach ($s in $summary) {
            Write-LogLine $LogPath "  $s" "Gray"
        }
        if ($p.ExitCode -ne 0 -or $failed.Count -gt 0 -or !$cudaPassed -or !$ranTests) {
            foreach ($f in ($failed | Select-Object -First 15)) {
                Write-LogLine $LogPath "  $f" "Red"
            }
            Write-LogLine $LogPath "[FAIL] CUDA self-test failed or ran no CUDA tests (exit code $($p.ExitCode)). Do not start the model; send the log file." "Red"
            return $false
        }
    }
    Write-LogLine $LogPath "[PASS] CUDA self-test" "Green"
    return $true
}

# the CPU backend of the package is built for AVX2 (Haswell level); an older CPU would crash with an illegal instruction
function Test-CpuSupport {
    param([string]$LogPath)
    try {
        Add-Type -Namespace LlamaGlm -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint feature);' -ErrorAction Stop
        # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
        if (![LlamaGlm.Cpu]::IsProcessorFeaturePresent(40)) {
            Write-LogLine $LogPath "[WARN] Windows reports no AVX2 on this CPU; the CPU backend needs AVX2 and may crash." "Yellow"
        }
    } catch {
        Write-LogLine $LogPath "[INFO] could not query CPU features ($($_.Exception.Message))" "DarkGray"
    }
}

function Set-DiagnosticEnv {
    param([string]$LogPath)
    # synchronous kernel launches so a bad kernel is reported at its own launch, no CUDA graphs
    $env:CUDA_LAUNCH_BLOCKING = "1"
    $env:GGML_CUDA_DISABLE_GRAPHS = "1"
    Write-LogLine $LogPath "[DIAG] CUDA_LAUNCH_BLOCKING=1, GGML_CUDA_DISABLE_GRAPHS=1 (slow, for finding a crashing kernel)" "Yellow"
}

function Clear-UnsafeEnv {
    param([string]$LogPath)
    foreach ($name in @("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "LLAMAGLM_LAYER_SPLIT", "LLAMA_LAYER_SPLIT")) {
        if (Test-Path "env:$name") {
            Remove-Item "env:$name"
            Write-LogLine $LogPath "[INFO] removed environment variable $name for this session" "DarkGray"
        }
    }
    # the launchers pass every llama.cpp setting on the command line; a leftover LLAMA_ARG_* variable
    # would change the server but not the Gate 2 projection
    foreach ($v in @(Get-ChildItem env: | Where-Object { $_.Name -like "LLAMA_ARG_*" })) {
        Remove-Item ("env:" + $v.Name)
        Write-LogLine $LogPath "[INFO] removed environment variable $($v.Name) for this session (use the launcher switches)" "DarkGray"
    }
}

# turn an opt-in environment variable on, or remove it so that an earlier session cannot leave it on
function Set-OptionalEnv {
    param([string]$LogPath, [string]$Name, [bool]$Enabled)
    if ($Enabled) {
        Set-Item "env:$Name" "1"
    } elseif (Test-Path "env:$Name") {
        Remove-Item "env:$Name"
        Write-LogLine $LogPath "[INFO] removed environment variable $Name for this session (option off)" "DarkGray"
    }
}

# -LockClocks: run gpu-clocks.ps1 as administrator, directly when this window already is one (an SSH session of an
# administrator account is), otherwise through a UAC prompt; returns $true if a clock was locked (or reset)
function Invoke-GpuClocks {
    param([string]$LogPath, [switch]$Reset)
    $script = Join-Path $PSScriptRoot "gpu-clocks.ps1"
    $out = [System.IO.Path]::GetTempFileName()
    $argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -OutFile `"$out`"" + $(if ($Reset) { " -Reset" } else { "" })
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $isAdmin = ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    try {
        if ($isAdmin) {
            Start-Process -FilePath "powershell" -ArgumentList $argLine -WindowStyle Hidden -Wait
        } else {
            Write-LogLine $LogPath "[INFO] GPU clocks need administrator rights: confirm the Windows prompt (or start this window as administrator)." "Yellow"
            Start-Process -FilePath "powershell" -ArgumentList $argLine -Verb RunAs -WindowStyle Hidden -Wait
        }
    } catch {
        Write-LogLine $LogPath "[WARN] GPU clocks: the administrator prompt was declined or is not available here; clocks unchanged." "Yellow"
    }
    # the result is read from the output file: an elevated process may not report its exit code back
    $lines = @(Get-Content -Path $out -ErrorAction SilentlyContinue)
    Remove-Item -Path $out -ErrorAction SilentlyContinue
    foreach ($l in $lines) {
        Write-LogLine $LogPath $l "Gray"
    }
    $summary = @($lines | Where-Object { $_ -match '^\[GPU CLOCKS\]' })
    if ($Reset) {
        return ($summary.Count -gt 0)
    }
    return (@($summary | Where-Object { $_ -match ': locked' }).Count -gt 0)
}
