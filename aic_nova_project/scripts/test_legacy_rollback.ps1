[CmdletBinding()]
param(
    [string]$LegacyDataRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$composePath = Join-Path $projectRoot "docker-compose.yml"
$rollbackPath = Join-Path $projectRoot "docker-compose.rollback.yml"
$legacyRoot = if ([string]::IsNullOrWhiteSpace($LegacyDataRoot)) {
    Join-Path $projectRoot "data"
} else {
    [System.IO.Path]::GetFullPath($LegacyDataRoot)
}
$legacyVolumes = @(
    "aic_nova_project_etcd_data",
    "aic_nova_project_minio_data",
    "aic_nova_project_milvus_data",
    "aic_nova_project_es_data"
)
$smokePath = Join-Path $legacyRoot "processed\online-encoder-smoke.json"
$previousLegacyRoot = [Environment]::GetEnvironmentVariable(
    "AIC_LEGACY_DATA_ROOT", "Process"
)

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

function Assert-LegacyVolumesExist {
    foreach ($volume in $legacyVolumes) {
        Invoke-CheckedCommand "docker" @("volume", "inspect", $volume)
    }
}

function Write-SmokeVectors {
    if ($DryRun) { return }
    $smoke = @{
        visual_features = @(1.0) + (@(0.0) * 511)
        ocr_features = @(1.0) + (@(0.0) * 767)
        asr_features = @(1.0) + (@(0.0) * 767)
        summary_features = @(1.0) + (@(0.0) * 767)
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $smokePath,
        ($smoke | ConvertTo-Json -Depth 3 -Compress),
        $encoding
    )
}

Push-Location $projectRoot
try {
    if (-not $DryRun -and -not (Test-Path -LiteralPath $legacyRoot -PathType Container)) {
        throw "Legacy data root not found: $legacyRoot"
    }
    [Environment]::SetEnvironmentVariable(
        "AIC_LEGACY_DATA_ROOT", $legacyRoot.Replace("\", "/"), "Process"
    )
    Assert-LegacyVolumesExist
    Invoke-CheckedCommand "docker" @("compose", "down", "--remove-orphans")
    Invoke-CheckedCommand "docker" @(
        "compose", "-f", $composePath, "-f", $rollbackPath,
        "up", "-d", "--wait",
        "etcd", "minio", "milvus-standalone", "elasticsearch"
    )
    $fingerprint = if ($DryRun) { "sha256:" + ("0" * 64) } else {
        $manifestPath = Join-Path $legacyRoot "processed\dataset-manifest.json"
        [string]((Get-Content -LiteralPath $manifestPath -Raw |
            ConvertFrom-Json).dataset_fingerprint)
    }
    Write-SmokeVectors
    Invoke-CheckedCommand "docker" @(
        "compose", "-f", $composePath, "-f", $rollbackPath,
        "run", "--rm",
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
    if ($DryRun) {
        Write-Host "COMMAND: docker compose -f $composePath -f $rollbackPath down --remove-orphans"
    } else {
        & docker compose -f $composePath -f $rollbackPath down --remove-orphans |
            Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Rollback stack cleanup failed with exit code $LASTEXITCODE"
        }
    }
    [Environment]::SetEnvironmentVariable(
        "AIC_LEGACY_DATA_ROOT", $previousLegacyRoot, "Process"
    )
    Pop-Location
}

Assert-LegacyVolumesExist
if ($DryRun) {
    Write-Host "Legacy rollback dry-run PASS; no Docker command was executed."
} else {
    Write-Host "Legacy rollback smoke PASS; all named volumes were preserved."
}
