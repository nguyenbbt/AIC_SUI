[CmdletBinding()]
param(
    [ValidateSet("Start", "Stop", "Status")]
    [string]$Action = "Status",
    [Parameter(Mandatory)][string]$CandidateRoot,
    [Parameter(Mandatory)][string]$BackupRoot,
    [string]$VqaBaseUrl = "",
    [string]$VqaApiKey = "",
    [ValidateRange(30, 900)][int]$StartupTimeoutSec = 300,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$uiRoot = Join-Path $projectRoot "ui"
$candidateRoot = [System.IO.Path]::GetFullPath($CandidateRoot)
$backupRoot = [System.IO.Path]::GetFullPath($BackupRoot)
$manifestPath = Join-Path $candidateRoot "processed\dataset-manifest.json"
$sqlitePath = Join-Path $candidateRoot "metadata.db"
$dataRoot = Join-Path $backupRoot "processed"
$runtimeRoot = Join-Path ([System.IO.Path]::GetTempPath()) "aic-nova-green-preview"
$statePath = Join-Path $runtimeRoot "processes.json"
$apiStdout = Join-Path $runtimeRoot "api.stdout.log"
$apiStderr = Join-Path $runtimeRoot "api.stderr.log"
$uiStdout = Join-Path $runtimeRoot "ui.stdout.log"
$uiStderr = Join-Path $runtimeRoot "ui.stderr.log"
$environmentBackup = @{}

function Write-Stage {
    param([Parameter(Mandatory)][string]$Name)
    Write-Host "`nSTAGE: $Name" -ForegroundColor Cyan
}

function Set-PreviewEnvironment {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Value
    )
    if (-not $environmentBackup.ContainsKey($Name)) {
        $environmentBackup[$Name] = [Environment]::GetEnvironmentVariable(
            $Name, "Process"
        )
    }
    $display = if ($Name -like "*_API_KEY") { "<redacted>" } else { $Value }
    Write-Host "ENV: ${Name}=${display}"
    if (-not $DryRun) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

function Restore-PreviewEnvironment {
    foreach ($entry in $environmentBackup.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            [string]$entry.Key, $entry.Value, "Process"
        )
    }
}

function Get-ListenerProcess {
    param([Parameter(Mandatory)][int]$Port)
    $listener = Get-NetTCPConnection `
        -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $listener) { return $null }
    return Get-CimInstance Win32_Process `
        -Filter "ProcessId=$($listener.OwningProcess)" `
        -ErrorAction SilentlyContinue
}

function Assert-PortAvailable {
    param([Parameter(Mandatory)][int]$Port)
    $process = Get-ListenerProcess -Port $Port
    if ($null -ne $process) {
        throw "Green preview port $Port is already owned by PID $($process.ProcessId)."
    }
}

function Wait-Until {
    param(
        [Parameter(Mandatory)][scriptblock]$Condition,
        [Parameter(Mandatory)][string]$Description
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSec)
    do {
        if (& $Condition) { return }
        Start-Sleep -Seconds 2
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out while waiting for $Description."
}

function Test-HttpReady {
    param([Parameter(Mandatory)][string]$Uri)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 10
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Get-ReadyManifest {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Green READY manifest is missing: $manifestPath"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.status -ne "READY") {
        throw "Green manifest status is not READY."
    }
    $fingerprint = [string]$manifest.dataset_fingerprint
    if ($fingerprint -notmatch '^sha256:[0-9a-f]{64}$') {
        throw "Green manifest fingerprint is invalid."
    }
    return $manifest
}

function Set-GreenRuntimeEnvironment {
    param([Parameter(Mandatory)][string]$Fingerprint)
    $vqaEnabled = -not [string]::IsNullOrWhiteSpace($VqaBaseUrl)
    $settings = [ordered]@{
        AIC_LOCAL_DATA_ROOT = $candidateRoot
        AIC_ONLINE_MILVUS_URI = "http://127.0.0.1:19531"
        AIC_ONLINE_ES_URI = "http://127.0.0.1:19200"
        AIC_ONLINE_SQLITE_PATH = $sqlitePath
        AIC_ONLINE_DATASET_MANIFEST_PATH = $manifestPath
        AIC_ONLINE_DATA_ROOT = $dataRoot
        AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT = $Fingerprint
        AIC_ONLINE_DATASET_MANIFEST_REQUIRED = "true"
        AIC_ONLINE_ENCODER_BACKEND = "modal"
        AIC_ONLINE_MODAL_ENCODER_APP = "aic-nova-online-encoders"
        AIC_ONLINE_MODAL_ENCODER_FUNCTION = "encode"
        AIC_ONLINE_MODAL_ENCODER_CACHE_SIZE = "256"
        AIC_ONLINE_RETRIEVAL_TIMEOUT_SEC = "180"
        AIC_ONLINE_QUERY_REWRITE_ENABLED = "false"
        AIC_ONLINE_TRAKE_ENABLED = "true"
        AIC_ONLINE_VQA_ENABLED = $vqaEnabled.ToString().ToLowerInvariant()
        AIC_ONLINE_QWEN_VLM_AUTO_CONFIGURE = $vqaEnabled.ToString().ToLowerInvariant()
        AIC_ONLINE_QWEN_VLM_BASE_URL = $VqaBaseUrl.TrimEnd("/")
        AIC_ONLINE_QWEN_VLM_API_KEY = $VqaApiKey
        AIC_ONLINE_QWEN_VLM_MODEL = "Qwen/Qwen3.5-4B"
        AIC_ONLINE_QWEN_VLM_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    }
    foreach ($entry in $settings.GetEnumerator()) {
        Set-PreviewEnvironment -Name $entry.Key -Value ([string]$entry.Value)
    }
}

function Stop-PreviewProcesses {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        Write-Host "Green preview has no owned process state."
        return
    }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    foreach ($processId in @([int]$state.ui_pid, [int]$state.api_pid)) {
        if ($processId -le 0 -or $DryRun) { continue }
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            & taskkill.exe /PID $processId /T /F | Out-Host
        }
    }
    if (-not $DryRun) {
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    }
}

function Start-Preview {
    $manifest = Get-ReadyManifest
    if (-not (Test-Path -LiteralPath $sqlitePath -PathType Leaf)) {
        throw "Green SQLite database is missing: $sqlitePath"
    }
    if (-not (Test-HttpReady -Uri "http://127.0.0.1:19200")) {
        throw "Green Elasticsearch is not reachable on port 19200."
    }
    if (-not (Test-HttpReady -Uri "http://127.0.0.1:19091/healthz")) {
        throw "Green Milvus is not ready on port 19091."
    }
    Assert-PortAvailable -Port 8001
    Assert-PortAvailable -Port 5174
    Set-GreenRuntimeEnvironment -Fingerprint ([string]$manifest.dataset_fingerprint
    )

    if ($DryRun) {
        Write-Host "COMMAND: $pythonExe -m uvicorn retrieval_api.main:app --host 127.0.0.1 --port 8001"
        Write-Host "COMMAND: npm.cmd run dev -- --host 127.0.0.1 --port 5174 --config vite.green.config.js"
        return
    }

    [System.IO.Directory]::CreateDirectory($runtimeRoot) | Out-Null
    foreach ($log in @($apiStdout, $apiStderr, $uiStdout, $uiStderr)) {
        if (Test-Path -LiteralPath $log) {
            Remove-Item -LiteralPath $log -Force
        }
    }
    $api = Start-Process -FilePath $pythonExe `
        -ArgumentList @(
            "-m", "uvicorn", "retrieval_api.main:app",
            "--host", "127.0.0.1", "--port", "8001"
        ) `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $apiStdout `
        -RedirectStandardError $apiStderr `
        -WindowStyle Hidden `
        -PassThru
    $ui = Start-Process -FilePath "npm.cmd" `
        -ArgumentList @(
            "run", "dev", "--", "--host", "127.0.0.1",
            "--port", "5174", "--strictPort", "--config",
            "vite.green.config.js"
        ) `
        -WorkingDirectory $uiRoot `
        -RedirectStandardOutput $uiStdout `
        -RedirectStandardError $uiStderr `
        -WindowStyle Hidden `
        -PassThru
    $state = [ordered]@{
        api_pid = $api.Id
        ui_pid = $ui.Id
        candidate_root = $candidateRoot
        dataset_fingerprint = [string]$manifest.dataset_fingerprint
        started_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    [System.IO.File]::WriteAllText(
        $statePath,
        ($state | ConvertTo-Json -Depth 4),
        [System.Text.UTF8Encoding]::new($false)
    )
    try {
        Wait-Until -Description "Green API readiness" -Condition {
            Test-HttpReady -Uri "http://127.0.0.1:8001/health/ready"
        }
        Wait-Until -Description "Green preview UI proxy" -Condition {
            Test-HttpReady -Uri "http://127.0.0.1:5174/api/health/ready"
        }
    } catch {
        Stop-PreviewProcesses
        throw
    }
    Write-Host "GREEN PREVIEW UI: http://127.0.0.1:5174" -ForegroundColor Green
    Write-Host "GREEN PREVIEW API: http://127.0.0.1:8001" -ForegroundColor Green
}

try {
    switch ($Action) {
        "Start" {
            Write-Stage "green-preview-start"
            Start-Preview
        }
        "Stop" {
            Write-Stage "green-preview-stop"
            Stop-PreviewProcesses
        }
        "Status" {
            Write-Stage "green-preview-status"
            foreach ($uri in @(
                "http://127.0.0.1:8001/health/ready",
                "http://127.0.0.1:5174/api/health/ready"
            )) {
                Write-Host "$uri -> $(if (Test-HttpReady -Uri $uri) { 'READY' } else { 'NOT READY' })"
            }
        }
    }
} finally {
    Restore-PreviewEnvironment
}
