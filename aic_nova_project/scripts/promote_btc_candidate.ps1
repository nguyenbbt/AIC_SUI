[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$CandidateRoot,
    [string]$StorageRoot = "",
    [string]$LegacyDataRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. "$(Join-Path $PSScriptRoot 'storage_paths.ps1')"
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$composePath = Join-Path $projectRoot "docker-compose.yml"
$rollbackPath = Join-Path $projectRoot "docker-compose.rollback.yml"
$envFile = Join-Path $projectRoot ".env"
$localDataRoot = Resolve-AicLocalDataRoot `
    -ExplicitStorageRoot $StorageRoot `
    -ProjectRoot $projectRoot `
    -EnvFile $envFile
$candidatePath = [System.IO.Path]::GetFullPath($CandidateRoot)
$legacyRoot = if ([string]::IsNullOrWhiteSpace($LegacyDataRoot)) {
    Join-Path $projectRoot "data"
} else {
    [System.IO.Path]::GetFullPath($LegacyDataRoot)
}
$previousLocalRoot = [Environment]::GetEnvironmentVariable(
    "AIC_LOCAL_DATA_ROOT", "Process"
)
$previousLegacyRoot = [Environment]::GetEnvironmentVariable(
    "AIC_LEGACY_DATA_ROOT", "Process"
)
$promoted = $false
$journalPath = ""

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    Write-Host "COMMAND: $FilePath $($Arguments -join ' ')"
    if ($DryRun) { return }
    & $FilePath @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Write-SmokeVectors {
    param([Parameter(Mandatory)][string]$Path)
    if ($DryRun) { return }
    $smoke = @{
        visual_features = @(1.0) + (@(0.0) * 511)
        ocr_features = @(1.0) + (@(0.0) * 767)
        asr_features = @(1.0) + (@(0.0) * 767)
        summary_features = @(1.0) + (@(0.0) * 767)
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $Path, ($smoke | ConvertTo-Json -Depth 3 -Compress), $encoding
    )
}

Push-Location $projectRoot
try {
    $datasetId = "dry-run-dataset"
    if (-not $DryRun) {
        Invoke-CheckedCommand $pythonExe @(
            "-m", "scripts.btc_storage_manager", "validate-stage",
            "--candidate-root", $candidatePath
        )
        $manifest = Get-Content -LiteralPath (
            Join-Path $candidatePath "processed\dataset-manifest.json"
        ) -Raw | ConvertFrom-Json
        $datasetId = [string]$manifest.dataset_id
    }
    $safeDatasetId = $datasetId -replace '[^A-Za-z0-9_.-]', '-'
    $journalPath = Join-Path $localDataRoot ".migration\publish-$safeDatasetId.json"

    Set-AicComposeDataRoot -Path $localDataRoot
    [Environment]::SetEnvironmentVariable(
        "AIC_LEGACY_DATA_ROOT", $legacyRoot.Replace("\", "/"), "Process"
    )
    Invoke-CheckedCommand "docker" @("compose", "down", "--remove-orphans")
    Invoke-CheckedCommand $pythonExe @(
        "-m", "scripts.btc_storage_manager", "promote",
        "--storage-root", $localDataRoot,
        "--candidate-root", $candidatePath,
        "--dataset-id", $datasetId
    )
    $promoted = $true
    Invoke-CheckedCommand "docker" @(
        "compose", "up", "-d", "--wait",
        "etcd", "minio", "milvus-standalone", "elasticsearch"
    )

    $manifestPath = Join-Path $localDataRoot "processed\dataset-manifest.json"
    $fingerprint = if ($DryRun) { "sha256:" + ("0" * 64) } else {
        [string]((Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json).dataset_fingerprint)
    }
    $smokePath = Join-Path $localDataRoot "processed\online-encoder-smoke.json"
    Write-SmokeVectors -Path $smokePath
    try {
        Invoke-CheckedCommand "docker" @(
            "compose", "run", "--rm",
            "-e", "AIC_ONLINE_MILVUS_URI=http://milvus-standalone:19530",
            "-e", "AIC_ONLINE_ES_URI=http://elasticsearch:9200",
            "-e", "AIC_ONLINE_SQLITE_PATH=/workspace/data/metadata.db",
            "-e", "AIC_ONLINE_DATASET_MANIFEST_PATH=/workspace/data/processed/dataset-manifest.json",
            "-e", "AIC_ONLINE_DATA_ROOT=/workspace/data/processed",
            "-e", "AIC_ONLINE_DATASET_MANIFEST_REQUIRED=true",
            "-e", "AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT=$fingerprint",
            "indexing", "python", "-m", "online.validate_contract",
            "--fail-on-partial", "--encoder-smoke-json",
            "/workspace/data/processed/online-encoder-smoke.json"
        )
    }
    finally {
        if (-not $DryRun -and (Test-Path -LiteralPath $smokePath -PathType Leaf)) {
            Remove-Item -LiteralPath $smokePath -Force
        }
    }
    Write-Host "Canonical E: promotion and revalidation PASS."
}
catch {
    if ($promoted -and -not $DryRun) {
        Invoke-CheckedCommand "docker" @("compose", "down", "--remove-orphans")
        Invoke-CheckedCommand $pythonExe @(
            "-m", "scripts.btc_storage_manager", "rollback",
            "--journal", $journalPath
        )
        Invoke-CheckedCommand "docker" @(
            "compose", "-f", $composePath, "-f", $rollbackPath,
            "up", "-d", "--wait",
            "etcd", "minio", "milvus-standalone", "elasticsearch"
        )
    }
    throw
}
finally {
    [Environment]::SetEnvironmentVariable(
        "AIC_LOCAL_DATA_ROOT", $previousLocalRoot, "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "AIC_LEGACY_DATA_ROOT", $previousLegacyRoot, "Process"
    )
    Pop-Location
}
