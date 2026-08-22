[CmdletBinding()]
param(
    [ValidateSet("Config", "Start", "Build", "Validate", "Status", "Stop")]
    [string]$Action = "Status",
    [Parameter(Mandatory)][string]$CandidateRoot,
    [Parameter(Mandatory)][string]$BackupRoot,
    [string]$DatasetId = "btc-full-873",
    [ValidateRange(1, 10000)][int]$BatchSize = 1000,
    [switch]$SkipPreview,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$composePath = Join-Path $projectRoot "docker-compose.green.yml"
$composeProject = "aic-nova-green"
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$previewScript = Join-Path $PSScriptRoot "run_green_preview.ps1"
$candidateRoot = [System.IO.Path]::GetFullPath($CandidateRoot)
$backupRoot = [System.IO.Path]::GetFullPath($BackupRoot)
$processedRoot = Join-Path $candidateRoot "processed"
$manifestPath = Join-Path $processedRoot "dataset-manifest.json"
$buildingManifestPath = Join-Path $processedRoot "dataset-manifest.building.json"
$smokePath = Join-Path $processedRoot "online-encoder-smoke.json"
$greenServices = @("etcd", "minio", "milvus-standalone", "elasticsearch")
$previousGreenRoot = [Environment]::GetEnvironmentVariable(
    "AIC_GREEN_DATA_ROOT", "Process"
)
$previousBackupRoot = [Environment]::GetEnvironmentVariable(
    "AIC_GREEN_BACKUP_ROOT", "Process"
)

function Write-Stage {
    param([Parameter(Mandatory)][string]$Name)
    Write-Host "`nSTAGE: $Name" -ForegroundColor Cyan
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    Write-Host "COMMAND: $FilePath $($Arguments -join ' ')"
    if ($DryRun) { return }
    & $FilePath @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}."
    }
}

function Invoke-GreenCompose {
    param([Parameter(Mandatory)][string[]]$Arguments)
    $composeArguments = @(
        "compose", "-f", $composePath, "-p", $composeProject
    ) + $Arguments
    Invoke-CheckedCommand "docker" $composeArguments
}

function Test-PathsOverlap {
    param(
        [Parameter(Mandatory)][string]$Left,
        [Parameter(Mandatory)][string]$Right
    )
    $leftPath = [System.IO.Path]::GetFullPath($Left).TrimEnd("\", "/")
    $rightPath = [System.IO.Path]::GetFullPath($Right).TrimEnd("\", "/")
    $comparison = [System.StringComparison]::OrdinalIgnoreCase
    if ($leftPath.Equals($rightPath, $comparison)) { return $true }
    return (
        $leftPath.StartsWith("${rightPath}\", $comparison) -or
        $rightPath.StartsWith("${leftPath}\", $comparison)
    )
}

function Assert-GreenRootDoesNotOverlapBlue {
    if ($DryRun) {
        Write-Host "CHECK: Green database root must not overlap Blue mounts."
        return
    }
    $inspectOutput = & "docker" @(
        "inspect", "aic_nova_milvus", "--format", "{{json .Mounts}}"
    ) 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($inspectOutput)) {
        Write-Host "Blue Milvus is absent; no active Blue mount to compare."
        return
    }
    $blueMounts = @($inspectOutput | ConvertFrom-Json)
    foreach ($mount in $blueMounts) {
        if ([string]$mount.Destination -ne "/var/lib/milvus") { continue }
        $blueDatabaseRoot = Split-Path -Parent ([string]$mount.Source)
        if (Test-PathsOverlap -Left $candidateRoot -Right $blueDatabaseRoot) {
            throw (
                "Green candidate root overlaps the active Blue database root: " +
                "$candidateRoot <-> $blueDatabaseRoot"
            )
        }
    }
}

function Assert-Inputs {
    foreach ($path in @($composePath, $pythonExe)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required file is unavailable: $path"
        }
    }
    if (-not (Test-Path -LiteralPath $backupRoot -PathType Container)) {
        throw "Backup root is unavailable: $backupRoot"
    }
    if (Test-PathsOverlap -Left $candidateRoot -Right $backupRoot) {
        throw "CandidateRoot and BackupRoot must not overlap."
    }
    $requiredSources = @(
        "metadata",
        "keyframes",
        "embeddings\visual",
        "module6-local\text_asr",
        "module6-local\text_ocr",
        "module6-local\text_summary",
        "asr-final\transcripts\transcripts",
        "asr-final\summaries\summaries",
        "ocr",
        "object_detection"
    )
    foreach ($relativePath in $requiredSources) {
        $source = Join-Path $backupRoot $relativePath
        if (-not (Test-Path -LiteralPath $source -PathType Container)) {
            throw "Required read-only source is unavailable: $source"
        }
    }
    Assert-GreenRootDoesNotOverlapBlue
}

function Initialize-GreenDirectories {
    if ($DryRun) {
        Write-Host "CREATE: $processedRoot and isolated Green database directories"
        return
    }
    [System.IO.Directory]::CreateDirectory($processedRoot) | Out-Null
    foreach ($name in @("etcd", "minio", "milvus", "elasticsearch")) {
        [System.IO.Directory]::CreateDirectory(
            (Join-Path $candidateRoot "databases\$name")
        ) | Out-Null
    }
}

function Write-SmokeVectors {
    if ($DryRun) { return }
    $vectors = [ordered]@{
        visual_features = @(1.0) + (@(0.0) * 511)
        ocr_features = @(1.0) + (@(0.0) * 767)
        asr_features = @(1.0) + (@(0.0) * 767)
        summary_features = @(1.0) + (@(0.0) * 767)
    }
    [System.IO.File]::WriteAllText(
        $smokePath,
        ($vectors | ConvertTo-Json -Depth 3 -Compress),
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Start-GreenInfrastructure {
    Initialize-GreenDirectories
    $arguments = @("up", "-d", "--build", "--wait") + $greenServices
    Invoke-GreenCompose $arguments
}

function Invoke-GreenValidation {
    if (-not $DryRun) {
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            throw "Green READY manifest is missing: $manifestPath"
        }
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        if ($manifest.status -ne "READY") {
            throw "Green manifest is not READY."
        }
        $fingerprint = [string]$manifest.dataset_fingerprint
        if ($fingerprint -notmatch '^sha256:[0-9a-f]{64}$') {
            throw "Green manifest fingerprint is invalid."
        }
    } else {
        $fingerprint = "sha256:" + ("0" * 64)
    }

    Write-SmokeVectors
    try {
        Invoke-GreenCompose @(
            "run", "--rm",
            "-e", "AIC_ONLINE_MILVUS_URI=http://milvus-standalone:19530",
            "-e", "AIC_ONLINE_ES_URI=http://elasticsearch:9200",
            "-e", "AIC_ONLINE_SQLITE_PATH=/workspace/data/metadata.db",
            "-e", "AIC_ONLINE_DATASET_MANIFEST_PATH=/workspace/data/processed/dataset-manifest.json",
            "-e", "AIC_ONLINE_DATA_ROOT=/workspace/data/processed",
            "-e", "AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT=$fingerprint",
            "-e", "AIC_ONLINE_DATASET_MANIFEST_REQUIRED=true",
            "indexing", "python", "-m", "online.validate_contract",
            "--fail-on-partial", "--encoder-smoke-json",
            "/workspace/data/processed/online-encoder-smoke.json"
        )
    } finally {
        if (-not $DryRun -and (Test-Path -LiteralPath $smokePath -PathType Leaf)) {
            Remove-Item -LiteralPath $smokePath -Force
        }
    }
}

Push-Location $projectRoot
try {
    Assert-Inputs
    [Environment]::SetEnvironmentVariable(
        "AIC_GREEN_DATA_ROOT", $candidateRoot.Replace("\", "/"), "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "AIC_GREEN_BACKUP_ROOT", $backupRoot.Replace("\", "/"), "Process"
    )

    switch ($Action) {
        "Config" {
            Write-Stage "green-config"
            Invoke-GreenCompose @("config")
        }
        "Start" {
            Write-Stage "green-infrastructure"
            Start-GreenInfrastructure
        }
        "Build" {
            Write-Stage "green-infrastructure"
            Start-GreenInfrastructure
            Write-Stage "green-indexing"
            Invoke-GreenCompose @("build", "indexing")
            Invoke-GreenCompose @(
                "run", "--rm", "indexing",
                "python", "-m", "src.indexing.cli",
                "--batch-size", [string]$BatchSize,
                "--force", "--reset-all", "--bulk-rebuild"
            )
            Write-Stage "green-publish-ready"
            Invoke-GreenCompose @(
                "run", "--rm", "indexing",
                "python", "-m", "src.indexing.publish_cli",
                "--data-dir", "/workspace/data/processed",
                "--dataset-id", $DatasetId,
                "--manifest-path", "/workspace/data/processed/dataset-manifest.json",
                "--building-manifest-path", "/workspace/data/processed/dataset-manifest.building.json"
            )
            Write-Stage "green-full-contract-validation"
            Invoke-GreenValidation
            Write-Host "GREEN READY: $manifestPath" -ForegroundColor Green
            if (-not $SkipPreview) {
                Write-Stage "green-preview"
                $previewArguments = @(
                    "-Action", "Start",
                    "-CandidateRoot", $candidateRoot,
                    "-BackupRoot", $backupRoot
                )
                if ($DryRun) { $previewArguments += "-DryRun" }
                Invoke-CheckedCommand $previewScript $previewArguments
            }
        }
        "Validate" {
            Write-Stage "green-full-contract-validation"
            Invoke-GreenValidation
        }
        "Status" {
            Write-Stage "green-status"
            Invoke-GreenCompose @("ps")
        }
        "Stop" {
            Write-Stage "green-stop"
            Invoke-GreenCompose @("down", "--remove-orphans")
        }
    }
} finally {
    [Environment]::SetEnvironmentVariable(
        "AIC_GREEN_DATA_ROOT", $previousGreenRoot, "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "AIC_GREEN_BACKUP_ROOT", $previousBackupRoot, "Process"
    )
    Pop-Location
}
