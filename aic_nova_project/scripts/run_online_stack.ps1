[CmdletBinding()]
param(
    [ValidateSet("Start", "Stop", "Status")]
    [string]$Action = "Start",
    [switch]$WithoutVQA,
    [string]$QwenVllmImage = "",
    [string]$VqaBaseUrl = "http://127.0.0.1:8001/v1",
    [string]$ModalProfile = "nguyenkhoanguyen2006",
    [switch]$SkipModalDeploy,
    [switch]$SkipContractValidation,
    [ValidateRange(30, 1800)]
    [int]$StartupTimeoutSec = 300,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$modalExe = Join-Path $projectRoot "venv\Scripts\modal.exe"
$uiRoot = Join-Path $projectRoot "ui"
$modalApp = Join-Path $projectRoot "scripts\online_modal_encoders.py"
$dataRoot = Join-Path $projectRoot "data\processed"
$manifestPath = Join-Path $dataRoot "dataset-manifest.json"
$sqlitePath = Join-Path $projectRoot "data\metadata.db"
$runtimeRoot = Join-Path ([System.IO.Path]::GetTempPath()) "aic-nova-online-runtime"
$statePath = Join-Path $runtimeRoot "processes.json"
$apiStdout = Join-Path $runtimeRoot "api.stdout.log"
$apiStderr = Join-Path $runtimeRoot "api.stderr.log"
$uiStdout = Join-Path $runtimeRoot "ui.stdout.log"
$uiStderr = Join-Path $runtimeRoot "ui.stderr.log"
$smokePath = Join-Path $runtimeRoot "modal-smoke-vectors.json"
$qwenContainerName = "aic-nova-qwen-vlm"
$qwenModel = "Qwen/Qwen3.5-4B"
$qwenRevision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
$dockerServices = @("etcd", "minio", "milvus-standalone", "elasticsearch")

function Write-Stage {
    param([Parameter(Mandatory)][string]$Name)
    Write-Host "`nSTAGE: $Name" -ForegroundColor Cyan
}

function Format-Command {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    return "$FilePath $($Arguments -join ' ')"
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $display = Format-Command $FilePath $Arguments
    Write-Host "COMMAND: $display"
    if ($DryRun) {
        return
    }
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $display"
    }
}

function Set-OnlineEnvironment {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Value
    )
    Write-Host "ENV: ${Name}=${Value}"
    if (-not $DryRun) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

function Get-ListenerProcess {
    param([Parameter(Mandatory)][int]$Port)
    $listener = Get-NetTCPConnection `
        -State Listen `
        -LocalPort $Port `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $listener) {
        return $null
    }
    return Get-CimInstance Win32_Process `
        -Filter "ProcessId=$($listener.OwningProcess)" `
        -ErrorAction SilentlyContinue
}

function Assert-PortAvailable {
    param([Parameter(Mandatory)][int]$Port)
    $process = Get-ListenerProcess -Port $Port
    if ($null -ne $process) {
        throw (
            "Port $Port is already owned by PID $($process.ProcessId): " +
            "$($process.CommandLine). Run -Action Stop or choose another port."
        )
    }
}

function Stop-OwnedListener {
    param(
        [Parameter(Mandatory)][int]$Port,
        [Parameter(Mandatory)][string]$ExpectedCommand
    )
    $process = Get-ListenerProcess -Port $Port
    if ($null -eq $process) {
        Write-Host "Port ${Port}: already closed"
        return
    }
    $commandLine = [string]$process.CommandLine
    if (
        $commandLine -notlike "*$ExpectedCommand*" -or
        $commandLine -notlike "*$projectRoot*"
    ) {
        throw (
            "Refusing to stop PID $($process.ProcessId) on port ${Port}; " +
            "it is not an AIC Nova $ExpectedCommand process."
        )
    }
    Stop-Process -Id $process.ProcessId -ErrorAction Stop
    Write-Host "Stopped PID $($process.ProcessId) on port $Port"
}

function Test-DockerReady {
    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        return $false
    }
    & docker info *> $null
    return $LASTEXITCODE -eq 0
}

function Wait-Until {
    param(
        [Parameter(Mandatory)][scriptblock]$Condition,
        [Parameter(Mandatory)][string]$Description,
        [int]$TimeoutSec = $StartupTimeoutSec
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
    do {
        if (& $Condition) {
            return
        }
        Start-Sleep -Seconds 2
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out after $TimeoutSec seconds while waiting for $Description."
}

function Start-DockerInfrastructure {
    if (-not (Test-DockerReady)) {
        $desktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (-not (Test-Path -LiteralPath $desktop -PathType Leaf)) {
            throw "Docker daemon is unavailable and Docker Desktop was not found."
        }
        Write-Host "Starting Docker Desktop..."
        Start-Process -FilePath $desktop -WindowStyle Hidden | Out-Null
        Wait-Until -Description "Docker daemon" -Condition { Test-DockerReady }
    }
    Invoke-CheckedCommand "docker" @(
        "compose", "up", "-d", "--wait",
        "etcd", "minio", "milvus-standalone", "elasticsearch"
    )
}

function Assert-Prerequisites {
    foreach ($path in @($pythonExe, $modalExe, $modalApp, $manifestPath, $sqlitePath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required file is missing: $path"
        }
    }
    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: docker"
    }
    if (-not (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: npm.cmd"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $uiRoot "node_modules\vite") -PathType Container)) {
        throw "UI dependencies are missing. Run: Set-Location ui; npm install"
    }
    Invoke-CheckedCommand $pythonExe @(
        "-c", "import fastapi, uvicorn, pymilvus, elasticsearch, modal"
    )
}

function Deploy-ModalEncoders {
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")
    if (-not $DryRun) {
        $currentProfile = (& $modalExe profile current | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to read the active Modal profile."
        }
        if ($currentProfile -notmatch [regex]::Escape($ModalProfile)) {
            throw "Modal profile mismatch. Expected '$ModalProfile', got '$currentProfile'."
        }
    } else {
        Write-Host "CHECK: Modal profile must be $ModalProfile"
    }
    if ($SkipModalDeploy) {
        Write-Host "SKIPPED: Modal deployment"
        return
    }
    Invoke-CheckedCommand $modalExe @("deploy", $modalApp)
}

function Test-QwenReady {
    try {
        $modelsUrl = "$($VqaBaseUrl.TrimEnd('/'))/models"
        $response = Invoke-RestMethod -Uri $modelsUrl -TimeoutSec 5
        return @($response.data | ForEach-Object { $_.id }) -contains $qwenModel
    } catch {
        return $false
    }
}

function Assert-PinnedVllmImage {
    if (
        [string]::IsNullOrWhiteSpace($QwenVllmImage) -or
        $QwenVllmImage -match ':latest$' -or
        $QwenVllmImage -notmatch '(@sha256:[0-9a-f]{64}$|:[^/:]+$)'
    ) {
        throw (
            "VQA needs a running Qwen endpoint or an explicit PINNED vLLM image " +
            "via -QwenVllmImage. The latest tag is rejected."
        )
    }
}

function Start-QwenIfNeeded {
    $modelsUrl = "$($VqaBaseUrl.TrimEnd('/'))/models"
    Write-Host "CHECK: $modelsUrl must serve $qwenModel"
    if ($DryRun) {
        if (-not [string]::IsNullOrWhiteSpace($QwenVllmImage)) {
            Assert-PinnedVllmImage
            Write-Host "COMMAND: docker run <pinned Qwen vLLM container> $QwenVllmImage"
        }
        return $false
    }
    if (Test-QwenReady) {
        Write-Host "Qwen VLM endpoint is ready: $VqaBaseUrl"
        return $false
    }
    Assert-PinnedVllmImage

    $existingImage = (& docker container inspect `
        --format "{{.Config.Image}}" $qwenContainerName 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) {
        if ($existingImage -ne $QwenVllmImage) {
            throw (
                "Container $qwenContainerName already exists with image " +
                "'$existingImage', expected '$QwenVllmImage'."
            )
        }
        Invoke-CheckedCommand "docker" @("start", $qwenContainerName) | Out-Host
    } else {
        Invoke-CheckedCommand "docker" @(
            "run", "-d", "--gpus", "all",
            "--name", $qwenContainerName,
            "--ipc=host",
            "-p", "127.0.0.1:8001:8000",
            "-v", "aic_nova_qwen_hf_cache:/root/.cache/huggingface",
            "-v", "aic_nova_qwen_vllm_cache:/root/.cache/vllm",
            $QwenVllmImage,
            $qwenModel,
            "--revision", $qwenRevision,
            "--served-model-name", $qwenModel,
            "--limit-mm-per-prompt", '{"image":12}'
        ) | Out-Host
    }
    Wait-Until -Description "Qwen VLM at $modelsUrl" -Condition { Test-QwenReady }
    return $true
}

function Set-RuntimeEnvironment {
    $vqaEnabled = if ($WithoutVQA) { "false" } else { "true" }
    $settings = [ordered]@{
        AIC_ONLINE_MILVUS_URI = "http://localhost:19530"
        AIC_ONLINE_ES_URI = "http://localhost:9200"
        AIC_ONLINE_SQLITE_PATH = $sqlitePath
        AIC_ONLINE_DATASET_MANIFEST_PATH = $manifestPath
        AIC_ONLINE_DATA_ROOT = $dataRoot
        AIC_ONLINE_DATASET_MANIFEST_REQUIRED = "true"
        AIC_ONLINE_ENCODER_BACKEND = "modal"
        AIC_ONLINE_MODAL_ENCODER_APP = "aic-nova-online-encoders"
        AIC_ONLINE_MODAL_ENCODER_FUNCTION = "encode"
        AIC_ONLINE_MODAL_ENVIRONMENT = ""
        AIC_ONLINE_MODAL_ENCODER_CACHE_SIZE = "256"
        AIC_ONLINE_RETRIEVAL_TIMEOUT_SEC = "180"
        AIC_ONLINE_TRAKE_ENABLED = "true"
        AIC_ONLINE_VQA_ENABLED = $vqaEnabled
        AIC_ONLINE_QWEN_VLM_AUTO_CONFIGURE = $vqaEnabled
        AIC_ONLINE_QWEN_VLM_BASE_URL = $VqaBaseUrl
        AIC_ONLINE_QWEN_VLM_MODEL = $qwenModel
        AIC_ONLINE_QWEN_VLM_REVISION = $qwenRevision
        AIC_ONLINE_QWEN_VLM_TIMEOUT_SEC = "15"
        AIC_ONLINE_QWEN_VLM_MAX_IMAGE_LONG_EDGE = "768"
    }
    foreach ($entry in $settings.GetEnumerator()) {
        Set-OnlineEnvironment -Name $entry.Key -Value ([string]$entry.Value)
    }
}

function Invoke-ContractValidation {
    if ($SkipContractValidation) {
        Write-Host "SKIPPED: read-only contract validation"
        return
    }
    Invoke-CheckedCommand $pythonExe @(
        "-m", "scripts.generate_online_modal_smoke_vectors",
        "--output", $smokePath
    )
    try {
        Invoke-CheckedCommand $pythonExe @(
            "-m", "online.validate_contract",
            "--fail-on-partial",
            "--encoder-smoke-json", $smokePath
        )
    } finally {
        if (-not $DryRun -and (Test-Path -LiteralPath $smokePath -PathType Leaf)) {
            Remove-Item -LiteralPath $smokePath -Force
        }
    }
}

function Start-ApiAndUi {
    if (-not $DryRun) {
        Assert-PortAvailable -Port 8000
        Assert-PortAvailable -Port 5173
        New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
        foreach ($logPath in @($apiStdout, $apiStderr, $uiStdout, $uiStderr)) {
            if (Test-Path -LiteralPath $logPath) {
                Remove-Item -LiteralPath $logPath -Force
            }
        }
    }

    Write-Stage "api"
    Write-Host "COMMAND: $pythonExe -m uvicorn retrieval_api.main:app --host 127.0.0.1 --port 8000"
    if ($DryRun) {
        Write-Stage "ui"
        Write-Host "COMMAND: npm.cmd run dev -- --host 127.0.0.1"
        return $null
    }
    $apiProcess = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @(
            "-m", "uvicorn", "retrieval_api.main:app",
            "--host", "127.0.0.1", "--port", "8000"
        ) `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $apiStdout `
        -RedirectStandardError $apiStderr `
        -WindowStyle Hidden `
        -PassThru

    Write-Stage "ui"
    Write-Host "COMMAND: npm.cmd run dev -- --host 127.0.0.1"
    $uiProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") `
        -WorkingDirectory $uiRoot `
        -RedirectStandardOutput $uiStdout `
        -RedirectStandardError $uiStderr `
        -WindowStyle Hidden `
        -PassThru

    return [pscustomobject]@{
        api_launcher_pid = $apiProcess.Id
        ui_launcher_pid = $uiProcess.Id
    }
}

function Save-ProcessState {
    param(
        [Parameter(Mandatory)]$Processes,
        [Parameter(Mandatory)][bool]$QwenStarted
    )
    if ($DryRun) {
        return
    }
    $state = [ordered]@{
        project_root = $projectRoot
        api_launcher_pid = $Processes.api_launcher_pid
        ui_launcher_pid = $Processes.ui_launcher_pid
        qwen_started_by_runner = $QwenStarted
        started_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    $json = $state | ConvertTo-Json
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($statePath, $json, $utf8WithoutBom)
}

function Show-RecentLogs {
    foreach ($path in @($apiStderr, $apiStdout, $uiStderr, $uiStdout)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Write-Warning "Last lines from $path"
            Get-Content -LiteralPath $path -Tail 30
        }
    }
}

function Wait-OnlineReadiness {
    $apiUrl = "http://127.0.0.1:8000/health/ready"
    Wait-Until -Description "FastAPI readiness" -Condition {
        try {
            $health = Invoke-RestMethod -Uri $apiUrl -TimeoutSec 10
            return $health.status -eq "ready"
        } catch {
            return $false
        }
    }
    $uiProxyUrl = "http://127.0.0.1:5173/api/health/ready"
    Wait-Until -Description "Vite UI and API proxy" -Condition {
        try {
            $health = Invoke-RestMethod -Uri $uiProxyUrl -TimeoutSec 10
            return $health.status -eq "ready"
        } catch {
            return $false
        }
    }
    Write-Host "Online stack is ready." -ForegroundColor Green
    Write-Host "UI:  http://127.0.0.1:5173"
    Write-Host "API: http://127.0.0.1:8000"
    Write-Host "Logs: $runtimeRoot"
}

function Stop-OnlineStack {
    Write-Stage "stop-api-ui"
    if ($DryRun) {
        Write-Host "CHECK/STOP: retrieval_api.main:app on port 8000"
        Write-Host "CHECK/STOP: vite on port 5173"
    } else {
        Stop-OwnedListener -Port 8000 -ExpectedCommand "retrieval_api.main:app"
        Stop-OwnedListener -Port 5173 -ExpectedCommand "vite"
    }

    Write-Stage "stop-qwen"
    $qwenStarted = $false
    if (-not $DryRun -and (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $qwenStarted = [bool]$state.qwen_started_by_runner
    }
    if ($qwenStarted) {
        Invoke-CheckedCommand "docker" @("stop", $qwenContainerName)
    } else {
        Write-Host "Qwen container was not recorded as runner-owned; leaving it unchanged."
    }

    Write-Stage "stop-docker-infrastructure"
    if ($DryRun -or (Test-DockerReady)) {
        Invoke-CheckedCommand "docker" @(
            "compose", "stop", "elasticsearch", "milvus-standalone", "minio", "etcd"
        )
    } else {
        Write-Warning "Docker daemon is unavailable; database containers could not be checked."
    }
    if (-not $DryRun -and (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        Remove-Item -LiteralPath $statePath -Force
    }
    Write-Host "Stopped Online services. Docker volumes and Modal deployment were preserved." -ForegroundColor Green
}

function Show-OnlineStatus {
    Write-Stage "status"
    if ($DryRun) {
        Write-Host "CHECK: ports 8000, 5173; Docker services; Qwen endpoint"
        return
    }
    foreach ($port in @(8000, 5173)) {
        $process = Get-ListenerProcess -Port $port
        if ($null -eq $process) {
            Write-Host "Port ${port}: CLOSED"
        } else {
            Write-Host "Port ${port}: LISTENING PID $($process.ProcessId)"
        }
    }
    if (Test-DockerReady) {
        Invoke-CheckedCommand "docker" @("compose", "ps", "-a")
    } else {
        Write-Host "Docker daemon: unavailable"
    }
    Write-Host "Qwen VLM: $(if (Test-QwenReady) { 'READY' } else { 'NOT READY' })"
    Write-Host "Modal deployment is intentionally retained and scales to zero when idle."
}

Push-Location $projectRoot
try {
    if ($Action -eq "Stop") {
        Stop-OnlineStack
        return
    }
    if ($Action -eq "Status") {
        Show-OnlineStatus
        return
    }

    Write-Stage "prerequisites"
    if ($DryRun) {
        Write-Host "CHECK: Python, Modal, Docker, npm, READY manifest and UI dependencies"
    } else {
        Assert-Prerequisites
        New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
    }

    Write-Stage "docker-infrastructure"
    if ($DryRun) {
        Invoke-CheckedCommand "docker" @(
            "compose", "up", "-d", "--wait",
            "etcd", "minio", "milvus-standalone", "elasticsearch"
        )
    } else {
        Start-DockerInfrastructure
    }

    Write-Stage "modal-encoder-deploy"
    Deploy-ModalEncoders

    $qwenStarted = $false
    if (-not $WithoutVQA) {
        Write-Stage "qwen-vlm"
        $qwenStarted = Start-QwenIfNeeded
    }

    Set-RuntimeEnvironment

    Write-Stage "contract-validation"
    Invoke-ContractValidation

    $processes = Start-ApiAndUi
    if (-not $DryRun) {
        Save-ProcessState -Processes $processes -QwenStarted $qwenStarted
    }

    Write-Stage "readiness"
    if ($DryRun) {
        Write-Host "CHECK: http://127.0.0.1:8000/health/ready"
        Write-Host "CHECK: http://127.0.0.1:5173/api/health/ready"
        Write-Host "UI: http://127.0.0.1:5173"
    } else {
        try {
            Wait-OnlineReadiness
        } catch {
            Show-RecentLogs
            throw
        }
    }
} catch {
    if ($Action -eq "Start" -and -not $DryRun) {
        Write-Warning "Startup failed. Run '.\scripts\run_online_stack.ps1 -Action Stop' to clean up safely."
    }
    throw
} finally {
    Pop-Location
}
