[CmdletBinding()]
param(
    [string]$StorageRoot = "",
    [string]$LegacyDataRoot = "",
    [Parameter(Mandatory)][ValidateCount(2, 2)][string[]]$ExpectedTestVideoIds,
    [switch]$AcceptSliceReplacement,
    [switch]$ConfirmDestructiveCleanup,
    [switch]$DeleteModalTestVolume,
    [string]$ModalEnvironment = "",
    [string]$ModalTestVolume = "aic-nova-offline-data",
    [string]$ModalInventoryPath = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. "$(Join-Path $PSScriptRoot 'storage_paths.ps1')"
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env"
$localDataRoot = Resolve-AicLocalDataRoot `
    -ExplicitStorageRoot $StorageRoot `
    -ProjectRoot $projectRoot `
    -EnvFile $envFile
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
$canonicalContainers = @(
    "aic_nova_etcd",
    "aic_nova_minio",
    "aic_nova_milvus",
    "aic_nova_elasticsearch"
)
$legacyIdsPath = Join-Path $localDataRoot ".migration\legacy-video-ids.json"
$previousLocalRoot = [Environment]::GetEnvironmentVariable(
    "AIC_LOCAL_DATA_ROOT", "Process"
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

function Assert-CanonicalBindMounts {
    if ($DryRun) {
        Write-Host "CHECK: canonical containers use bind mounts rooted at $localDataRoot"
        return
    }
    foreach ($container in $canonicalContainers) {
        $inspection = @(& docker inspect $container | ConvertFrom-Json)
        if ($LASTEXITCODE -ne 0 -or $inspection.Count -ne 1) {
            throw "Canonical container is unavailable: $container"
        }
        $mounts = @($inspection[0].Mounts)
        $dataMounts = @($mounts | Where-Object {
            $_.Destination -in @(
                "/etcd", "/minio_data", "/var/lib/milvus",
                "/usr/share/elasticsearch/data"
            )
        })
        if ($dataMounts.Count -ne 1 -or $dataMounts[0].Type -ne "bind") {
            throw "$container is not using the expected canonical bind mount"
        }
        $source = [System.IO.Path]::GetFullPath([string]$dataMounts[0].Source)
        if (-not $source.StartsWith(
            $localDataRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "$container mount is outside the configured SSD root: $source"
        }
    }
}

function Assert-CanonicalDatasetGate {
    $manifestPath = Join-Path $localDataRoot "processed\dataset-manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Canonical READY manifest is missing: $manifestPath"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.status -ne "READY") {
        throw "Canonical dataset is not READY."
    }
    $videoCount = [int64]$manifest.record_counts.videos
    if (-not $AcceptSliceReplacement -and $videoCount -ne 873) {
        throw "Cleanup requires full 873-video PASS; manifest has $videoCount."
    }
    Write-Host "CANONICAL DATASET: $($manifest.dataset_id), videos=$videoCount"
}

function Assert-LegacyIds {
    if ($DryRun) {
        Write-Host "CHECK: legacy SQLite contains exactly: $($ExpectedTestVideoIds -join ', ')"
        return
    }
    Invoke-CheckedCommand $pythonExe @(
        "-m", "scripts.btc_storage_manager", "sqlite-video-ids",
        "--database", (Join-Path $legacyRoot "metadata.db"),
        "--output", $legacyIdsPath
    )
    $actual = @((Get-Content -LiteralPath $legacyIdsPath -Raw |
        ConvertFrom-Json).video_ids | Sort-Object)
    $expected = @($ExpectedTestVideoIds | Sort-Object)
    if ($actual.Count -ne 2 -or (Compare-Object $actual $expected)) {
        throw "Legacy SQLite IDs do not exactly match the two approved test IDs."
    }
}

function Assert-ModalCleanupGate {
    if (-not $DeleteModalTestVolume) { return }
    if ([string]::IsNullOrWhiteSpace($ModalEnvironment)) {
        throw "-ModalEnvironment is required when deleting a Modal volume."
    }
    if ([string]::IsNullOrWhiteSpace($ModalInventoryPath)) {
        throw "-ModalInventoryPath is required to verify the remote test IDs."
    }
    $inventoryPath = [System.IO.Path]::GetFullPath($ModalInventoryPath)
    $inventory = Get-Content -LiteralPath $inventoryPath -Raw | ConvertFrom-Json
    $remoteIds = @($inventory.files | ForEach-Object {
        [System.IO.Path]::GetFileNameWithoutExtension([string]$_.relative_path)
    } | Sort-Object -Unique)
    $expected = @($ExpectedTestVideoIds | Sort-Object)
    if ($remoteIds.Count -ne 2 -or (Compare-Object $remoteIds $expected)) {
        throw "Modal inventory does not contain exactly the approved test IDs."
    }
    if (-not $DryRun) {
        $volumeList = & modal volume list -e $ModalEnvironment --json |
            ConvertFrom-Json
        if ($LASTEXITCODE -ne 0 -or $ModalTestVolume -notin @(
            $volumeList | ForEach-Object { $_.name }
        )) {
            throw "Modal test volume is absent from the selected environment."
        }
    }
}

function Assert-ZeroVolumeReferences {
    foreach ($volume in $legacyVolumes) {
        if ($DryRun) {
            Write-Host "CHECK: zero container references for $volume"
            continue
        }
        $references = @(& docker ps -a --filter "volume=$volume" --format "{{.ID}}")
        if ($LASTEXITCODE -ne 0 -or $references.Count -gt 0) {
            throw "Legacy volume still has container references: $volume"
        }
    }
}

function Remove-ApprovedLegacyPaths {
    $allowedRoot = [System.IO.Path]::GetFullPath($legacyRoot).TrimEnd("\") + "\"
    $targets = @(
        (Join-Path $legacyRoot "raw_videos"),
        (Join-Path $legacyRoot "processed"),
        (Join-Path $legacyRoot "metadata.db"),
        (Join-Path $legacyRoot "metadata.db-shm"),
        (Join-Path $legacyRoot "metadata.db-wal")
    )
    foreach ($target in $targets) {
        $resolved = [System.IO.Path]::GetFullPath($target)
        if (-not $resolved.StartsWith(
            $allowedRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Cleanup target escaped the legacy data root: $resolved"
        }
        Write-Host "DELETE LOCAL: $resolved"
        if (-not $DryRun -and (Test-Path -LiteralPath $resolved)) {
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}

Push-Location $projectRoot
try {
    Assert-CanonicalDatasetGate
    Assert-CanonicalBindMounts
    Assert-LegacyIds
    Assert-ModalCleanupGate

    Write-Host "LEGACY VOLUMES TO DELETE:"
    $legacyVolumes | ForEach-Object { Write-Host "  $_" }
    Write-Host "LEGACY LOCAL ROOT: $legacyRoot"
    if (-not $ConfirmDestructiveCleanup) {
        throw "Destructive gate closed. Re-run with -ConfirmDestructiveCleanup."
    }

    Set-AicComposeDataRoot -Path $localDataRoot
    Invoke-CheckedCommand "docker" @("compose", "down", "--remove-orphans")
    Assert-ZeroVolumeReferences
    foreach ($volume in $legacyVolumes) {
        Invoke-CheckedCommand "docker" @("volume", "rm", $volume)
    }
    Remove-ApprovedLegacyPaths
    if ($DeleteModalTestVolume) {
        Invoke-CheckedCommand "modal" @(
            "volume", "delete", "-y", "-e", $ModalEnvironment,
            $ModalTestVolume
        )
    }
    Invoke-CheckedCommand "docker" @(
        "compose", "up", "-d", "--wait",
        "etcd", "minio", "milvus-standalone", "elasticsearch"
    )
    Write-Host "Cleanup complete. Run canonical full contract and Online smoke tests."
}
finally {
    [Environment]::SetEnvironmentVariable(
        "AIC_LOCAL_DATA_ROOT", $previousLocalRoot, "Process"
    )
    Pop-Location
}
