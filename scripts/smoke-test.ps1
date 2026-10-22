# LlamaGLM acceptance gates 3-5. Run on Stage A in a second window after start-stage-a.ps1
# (which already ran gates 0-2). Stops at the first failure.
#   Gate 3  model loaded: /health ok AND the server log shows weights on all 4 RPC devices and mmap
#   Gate 4  real tokens: raw /completion and chat completion return tokens, and the answers are right (Paris, 4)
#   Gate 5  stability: two more requests, health still ok, token counters advance
param (
    [string]$Endpoint = "",
    [int]$LoadTimeoutMin = 60,
    [int]$RequestTimeoutSec = 1800,
    [ValidateSet("3", "4", "5", "all")][string]$Gate = "all",
    [double]$MinRpcBufferMiB = 100,
    # skip the check that the answers mention Paris and 4 (only if the printed text is clearly correct)
    [switch]$NoAnswerCheck,
    [string]$ConfigPath = (Join-Path $PSScriptRoot "cluster_config.json")
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "llamaglm-common.ps1")
$cfg = Get-LlamaGlmConfig $ConfigPath
if ([string]::IsNullOrEmpty($Endpoint)) {
    $Endpoint = "http://{0}:{1}" -f $cfg.stage_a.api_host, $cfg.stage_a.api_port
}
$log = New-LlamaGlmLogPath "smoke-test"
Write-LogLine $log "LlamaGLM $($cfg.version) smoke test against $Endpoint, log: $log" "Cyan"

function Stop-Gate([string]$Msg) {
    Write-LogLine $log "[FAIL] $Msg" "Red"
    Write-LogLine $log "Collect logs with .\collect-logs.ps1 and keep the worker windows open." "Yellow"
    exit 1
}

function Invoke-Api([string]$Path, $Body) {
    $json = $Body | ConvertTo-Json -Depth 6 -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$Endpoint$Path" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec $RequestTimeoutSec
    $sw.Stop()
    Write-LogLine $log ("  {0} answered in {1:N1} s" -f $Path, $sw.Elapsed.TotalSeconds) "Gray"
    return $r
}

function Get-TokensPredicted {
    try {
        $m = Invoke-WebRequest -Uri "$Endpoint/metrics" -UseBasicParsing -TimeoutSec 30
        $line = ($m.Content -split "`n") | Where-Object { $_ -match "^llamacpp:tokens_predicted_total" } | Select-Object -First 1
        if ($line) {
            return [double]($line -split "\s+")[1]
        }
    } catch {
    }
    return -1
}

# ---- Gate 3 ---------------------------------------------------------------------------------------------
Write-LogLine $log "[GATE 3] waiting for the model to load (first load can take 15-30 min)" "Cyan"
$deadline = (Get-Date).AddMinutes($LoadTimeoutMin)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-RestMethod -Uri "$Endpoint/health" -TimeoutSec 10
        if ($h.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {
        # 503 while loading, connection refused before the listener is up
    }
    Start-Sleep -Seconds 10
}
if (!$ready) {
    Stop-Gate "server did not report /health ok within $LoadTimeoutMin min (look at the llama-server window for the error)"
}
Write-LogLine $log "  /health ok" "Green"

$serverLog = Get-ChildItem -Path (Join-Path $PSScriptRoot "logs") -Filter "llama-server-*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (!$serverLog) {
    Stop-Gate "no logs\llama-server-*.log found; start the server with start-stage-a.ps1"
}
$text = Get-Content -Path $serverLog.FullName
$placement = @($text | Where-Object { $_ -match "load_tensors: layers|load_tensors: output layer|model buffer size|load_mode =" })
foreach ($l in $placement) {
    Write-LogLine $log "  $l" "Gray"
}
if (!($text | Where-Object { $_ -match "load_mode = mmap" })) {
    Stop-Gate "the model was not loaded with mmap; Stage A RAM would be overcommitted"
}
# buffer type names are RPC<device on that worker>[host:port]; the real load is logged last
$rpcBuffers = @($text | Where-Object { $_ -match "RPC[01]\[[^\]]+\] model buffer size =\s+([0-9.]+) MiB" } | Select-Object -Last 4 | ForEach-Object {
    [void]($_ -match "RPC[01]\[[^\]]+\] model buffer size =\s+([0-9.]+) MiB"); [double]$Matches[1] })
if ($rpcBuffers.Count -lt 4 -or @($rpcBuffers | Where-Object { $_ -lt $MinRpcBufferMiB }).Count -gt 0) {
    Stop-Gate ("expected 4 RPC weight buffers of at least $MinRpcBufferMiB MiB, found: {0}" -f ($rpcBuffers -join ", "))
}
Write-LogLine $log ("[PASS] Gate 3: model loaded, RPC weight buffers (MiB): {0}" -f ($rpcBuffers -join ", ")) "Green"
if ($Gate -eq "3") {
    exit 0
}

# ---- Gate 4 ---------------------------------------------------------------------------------------------
Write-LogLine $log "[GATE 4] first tokens (raw completion, then chat)" "Cyan"
$before = Get-TokensPredicted
try {
    $r = Invoke-Api "/completion" @{ prompt = "The capital of France is"; n_predict = 16; temperature = 0 }
} catch {
    Stop-Gate "raw /completion failed: $_"
}
Write-LogLine $log ("  raw: tokens_predicted={0} stop_type={1} text='{2}'" -f $r.tokens_predicted, $r.stop_type, $r.content) "White"
if ($r.tokens_predicted -le 0 -or [string]::IsNullOrWhiteSpace($r.content)) {
    Stop-Gate "raw completion produced no tokens"
}
# broken kernels still produce tokens, but not the right answer
if (!$NoAnswerCheck -and $r.content -notmatch "Paris") {
    Stop-Gate "raw completion does not mention Paris (text above). Wrong output points to broken kernels: compare with -BinDir (FIRST-RUN.md section 6)."
}

# the GLM template always opens <think>; keep reasoning short and accept text in reasoning_content
$chat = @{
    model = $cfg.model.alias
    messages = @(@{ role = "user"; content = "Reply with one short sentence: what is 2 + 2?" })
    max_tokens = 256
    temperature = 0
    chat_template_kwargs = @{ reasoning_effort = "low" }
}
try {
    $c = Invoke-Api "/v1/chat/completions" $chat
} catch {
    Stop-Gate "chat completion failed: $_"
}
$msg = $c.choices[0].message
Write-LogLine $log ("  chat: completion_tokens={0} finish_reason={1}" -f $c.usage.completion_tokens, $c.choices[0].finish_reason) "White"
Write-LogLine $log ("  reasoning: '{0}'" -f $msg.reasoning_content) "DarkGray"
Write-LogLine $log ("  content:   '{0}'" -f $msg.content) "White"
if ($c.usage.completion_tokens -le 0) {
    Stop-Gate "chat completion_tokens = $($c.usage.completion_tokens)"
}
if ([string]::IsNullOrWhiteSpace($msg.content) -and [string]::IsNullOrWhiteSpace($msg.reasoning_content)) {
    Stop-Gate "chat completion returned no text"
}
if (@("stop", "length") -notcontains $c.choices[0].finish_reason) {
    Stop-Gate "unexpected finish_reason '$($c.choices[0].finish_reason)'"
}
if (!$NoAnswerCheck -and ("{0} {1}" -f $msg.content, $msg.reasoning_content) -notmatch "\b4\b|\bfour\b") {
    Stop-Gate "chat answer to 2 + 2 contains no 4 (text above). Wrong output points to broken kernels: compare with -BinDir (FIRST-RUN.md section 6)."
}
$what = if ($NoAnswerCheck) { "tokens (answers not checked)" } else { "correct answers" }
Write-LogLine $log "[PASS] Gate 4: the model generated $what across the cluster (weights for 4 RPC devices exist only on the workers)" "Green"
if ($Gate -eq "4") {
    exit 0
}

# ---- Gate 5 ---------------------------------------------------------------------------------------------
Write-LogLine $log "[GATE 5] two more requests" "Cyan"
foreach ($q in @("Name three primary colors.", "Count from 1 to 5.")) {
    $chat.messages = @(@{ role = "user"; content = $q })
    try {
        $c = Invoke-Api "/v1/chat/completions" $chat
    } catch {
        Stop-Gate "request '$q' failed: $_"
    }
    Write-LogLine $log ("  '{0}' -> {1} tokens, content '{2}'" -f $q, $c.usage.completion_tokens, $c.choices[0].message.content) "White"
    if ($c.usage.completion_tokens -le 0) {
        Stop-Gate "request '$q' produced no tokens"
    }
}
$h = Invoke-RestMethod -Uri "$Endpoint/health" -TimeoutSec 10
if ($h.status -ne "ok") {
    Stop-Gate "health after requests: '$($h.status)'"
}
$after = Get-TokensPredicted
if ($before -ge 0 -and $after -le $before) {
    Stop-Gate "tokens_predicted_total did not advance ($before -> $after)"
}
Write-LogLine $log ("[PASS] Gate 5: health ok, tokens_predicted_total {0} -> {1}" -f $before, $after) "Green"
Write-LogLine $log "Check the worker windows: their [stats] lines must show 'graphs' increasing on both devices." "Yellow"
exit 0
