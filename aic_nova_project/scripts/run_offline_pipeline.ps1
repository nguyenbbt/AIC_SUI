[CmdletBinding()]
param(
    [string]$VolumeName = "aic-nova-btc-data",
    [string]$StorageRoot = "",
    [switch]$Force,
    [switch]$ResetIndex,
    [switch]$SkipIndex,
    [switch]$SkipUpload,
    [switch]$ForceUpload,
    [switch]$ApprovePaidUpload,
    [switch]$DryRun,
    [string]$DatasetId = "",
    [string]$SliceVideoId = "",
    [string]$ModalEnvironment = "",
    [switch]$IndexStagedOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. "$(Join-Path $PSScriptRoot 'storage_paths.ps1')"
$envFile = Join-Path $projectRoot ".env"
$previousLocalDataRoot = [Environment]::GetEnvironmentVariable(
    "AIC_LOCAL_DATA_ROOT", "Process"
)
$localData = Resolve-AicLocalDataRoot `
    -ExplicitStorageRoot $StorageRoot `
    -ProjectRoot $projectRoot `
    -EnvFile $envFile
$modalRunner = Join-Path $projectRoot "scripts\offline_modal_runner.py"
$localRawVideos = Join-Path $localData "raw_videos"
$activeRawVideos = $localRawVideos
$localCaptions = Join-Path $localData "captions"
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$resolvedDatasetId = if ([string]::IsNullOrWhiteSpace($DatasetId)) {
    "aic2026-btc-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))"
} else {
    $DatasetId
}
$safeDatasetId = $resolvedDatasetId -replace '[^A-Za-z0-9_.-]', '-'
$stagingRoot = Join-Path $localData ".staging"
$stagingParent = Join-Path $stagingRoot "dataset-$resolvedDatasetId"
$localProcessed = Join-Path $stagingParent "processed"
$migrationRoot = Join-Path $localData ".migration"
$localInventoryPath = Join-Path $migrationRoot "modal-local-$safeDatasetId.json"
$remoteInventoryPath = Join-Path $migrationRoot "modal-remote-$safeDatasetId.json"
$uploadPlanPath = Join-Path $migrationRoot "modal-upload-plan-$safeDatasetId.json"
$onlineSmokePath = Join-Path $localProcessed "online-encoder-smoke.json"
$previousPythonUtf8 = [Environment]::GetEnvironmentVariable(
    "PYTHONUTF8", "Process"
)
$previousPythonIoEncoding = [Environment]::GetEnvironmentVariable(
    "PYTHONIOENCODING", "Process"
)
$previousModalDataVolume = [Environment]::GetEnvironmentVariable(
    "AIC_MODAL_DATA_VOLUME", "Process"
)
$previousModalEnvironment = [Environment]::GetEnvironmentVariable(
    "MODAL_ENVIRONMENT", "Process"
)
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")
[Environment]::SetEnvironmentVariable(
    "AIC_MODAL_DATA_VOLUME", $VolumeName, "Process"
)
if (-not [string]::IsNullOrWhiteSpace($ModalEnvironment)) {
    [Environment]::SetEnvironmentVariable(
        "MODAL_ENVIRONMENT", $ModalEnvironment, "Process"
    )
}
$candidateDockerStarted = $false

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

    $displayCommand = Format-Command $FilePath $Arguments
    Write-Host "COMMAND: $displayCommand"
    if ($DryRun) {
        return
    }

    & $FilePath @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $displayCommand"
    }
}

function Assert-Prerequisites {
    if ($DryRun) {
        return
    }
    $requiredCommands = @("docker")
    if (-not $IndexStagedOnly) {
        $requiredCommands += "modal"
    }
    foreach ($commandName in $requiredCommands) {
        if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
            throw "Required command is not available: $commandName"
        }
    }
    if (-not (Test-Path -LiteralPath $modalRunner -PathType Leaf)) {
        throw "Modal runner not found: $modalRunner"
    }
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        throw "Project Python not found: $pythonExe"
    }
    if ($IndexStagedOnly) {
        if (-not (Test-Path -LiteralPath $localProcessed -PathType Container)) {
            throw "Staged processed directory not found: $localProcessed"
        }
        return
    }
    if ($SkipUpload -and $ForceUpload) {
        throw "-SkipUpload and -ForceUpload cannot be used together."
    }

    $videoFiles = @(
        Get-ChildItem -LiteralPath $localRawVideos -File -Recurse |
            Where-Object { $_.Extension -in @(".mp4", ".mkv", ".avi", ".webm") }
    )
    if ($videoFiles.Count -eq 0) {
        throw "No input videos found in $localRawVideos"
    }

    $requiredGiB = if ([string]::IsNullOrWhiteSpace($SliceVideoId)) {
        251.85
    } else {
        20.0
    }
    $driveRoot = [System.IO.Path]::GetPathRoot($localData)
    $drive = [System.IO.DriveInfo]::new($driveRoot)
    $freeGiB = $drive.AvailableFreeSpace / 1GB
    Write-Host (
        "FREE SPACE: {0:N3} GiB; required policy: {1:N2} GiB" -f
        $freeGiB, $requiredGiB
    )
    if ($freeGiB -lt $requiredGiB) {
        throw "Configured storage root does not meet the free-space policy."
    }
    Invoke-CheckedCommand "modal" @("profile", "current")
}

function Initialize-SliceInput {
    if ([string]::IsNullOrWhiteSpace($SliceVideoId)) {
        return $localRawVideos
    }
    $sliceRoot = Join-Path (
        Join-Path $localData ".staging\inputs"
    ) "dataset-$resolvedDatasetId\raw_videos"
    if ($DryRun) {
        Write-Host "SLICE INPUT: $SliceVideoId -> $sliceRoot (hardlink preferred)"
        return $sliceRoot
    }
    $matches = @(
        Get-ChildItem -LiteralPath $localRawVideos -File -Recurse |
            Where-Object {
                $_.Extension -in @(".mp4", ".mkv", ".avi", ".webm") -and
                $_.BaseName -eq $SliceVideoId
            }
    )
    if ($matches.Count -ne 1) {
        throw "Slice video_id must match exactly one file; found $($matches.Count)."
    }
    $rawPrefix = $localRawVideos.TrimEnd("\") + "\"
    $relativePath = $matches[0].FullName.Substring($rawPrefix.Length)
    $target = Join-Path $sliceRoot $relativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) |
        Out-Null
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        if ((Get-Item -LiteralPath $target).Length -ne $matches[0].Length) {
            throw "Existing slice input has a different byte size: $target"
        }
    } else {
        try {
            New-Item -ItemType HardLink -Path $target `
                -Target $matches[0].FullName | Out-Null
        }
        catch {
            Write-Warning "Hardlink unavailable; copying one slice video on E:."
            Copy-Item -LiteralPath $matches[0].FullName -Destination $target
        }
    }
    Write-Host "SLICE INPUT: $SliceVideoId -> $target"
    return $sliceRoot
}

function Ensure-ModalVolume {
    if ($DryRun) {
        Invoke-CheckedCommand "modal" @("volume", "create", $VolumeName)
        return
    }

    $volumeJson = & modal volume list --json
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list Modal volumes."
    }
    $volumes = $volumeJson | ConvertFrom-Json
    $volumeNames = @($volumes | ForEach-Object { $_.name })
    if ($VolumeName -notin $volumeNames) {
        Invoke-CheckedCommand "modal" @("volume", "create", $VolumeName)
    } else {
        Write-Host "Modal volume already exists: $VolumeName"
    }
}

function Assert-ModalUploadApproved {
    param([Parameter(Mandatory)][long]$ChangedBytes)
    Write-Host "PROJECTED UPLOAD BYTES: $ChangedBytes"
    if ($ChangedBytes -gt 0 -and -not $ApprovePaidUpload -and -not $DryRun) {
        throw (
            "Modal upload may incur storage and transfer cost. " +
            "Re-run with -ApprovePaidUpload after reviewing projected bytes."
        )
    }
    Write-Host "UPLOAD MODE: $(if ($ForceUpload) { 'force' } else { 'resume' })"
}

function New-LocalInventory {
    if ($DryRun) {
        return [pscustomobject]@{
            video_count = 0
            total_bytes = 0
            aggregate_sha256 = ("0" * 64)
        }
    }
    New-Item -ItemType Directory -Force -Path $migrationRoot | Out-Null
    Invoke-CheckedCommand $pythonExe @(
        "-m", "scripts.btc_storage_manager", "inventory",
        "--root", $activeRawVideos,
        "--output", $localInventoryPath
    )
    return Get-Content -LiteralPath $localInventoryPath -Raw | ConvertFrom-Json
}

function Get-RemoteInventory {
    if ($DryRun) {
        return $null
    }
    if (Test-Path -LiteralPath $remoteInventoryPath -PathType Leaf) {
        Remove-Item -LiteralPath $remoteInventoryPath -Force
    }
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & modal volume get $VolumeName "/manifests/raw-inventory.json" `
            $remoteInventoryPath 2>$null | Out-Host
        $remoteGetExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if (
        $remoteGetExitCode -ne 0 -or
        -not (Test-Path -LiteralPath $remoteInventoryPath)
    ) {
        Write-Host "REMOTE INVENTORY: absent; planning an initial upload."
        return $null
    }
    return Get-Content -LiteralPath $remoteInventoryPath -Raw | ConvertFrom-Json
}

function New-UploadPlan {
    param([Parameter(Mandatory)][AllowNull()]$RemoteInventory)
    if ($DryRun) {
        return [pscustomobject]@{ changed_count = 0; changed_bytes = 0; files = @() }
    }
    $arguments = @(
        "-m", "scripts.btc_storage_manager", "changed",
        "--local", $localInventoryPath,
        "--output", $uploadPlanPath
    )
    if (-not $ForceUpload -and $null -ne $RemoteInventory) {
        $arguments += @("--remote", $remoteInventoryPath)
    }
    Invoke-CheckedCommand $pythonExe $arguments
    return Get-Content -LiteralPath $uploadPlanPath -Raw | ConvertFrom-Json
}

function Invoke-ResumableUpload {
    param([Parameter(Mandatory)]$UploadPlan)
    foreach ($entry in @($UploadPlan.files)) {
        $relativePath = [string]$entry.relative_path
        $localRelativePath = $relativePath.Replace(
            "/", [string][System.IO.Path]::DirectorySeparatorChar
        )
        $source = Join-Path $activeRawVideos $localRelativePath
        Invoke-CheckedCommand "modal" @(
            "volume", "put", "--force", $VolumeName,
            $source, "/raw_videos/$relativePath"
        )
    }
    if (-not $DryRun) {
        Invoke-CheckedCommand "modal" @(
            "volume", "put", "--force", $VolumeName,
            $localInventoryPath, "/manifests/raw-inventory.json"
        )
    }
}

function Assert-RemoteInventory {
    param([Parameter(Mandatory)]$LocalInventory)
    Write-Host "CHECK: remote /raw_videos count, bytes and aggregate SHA-256"
    Invoke-CheckedCommand "modal" @(
        "run", $modalRunner,
        "--verify-root", "/data/raw_videos",
        "--expected-count", ([string]$LocalInventory.video_count),
        "--expected-bytes", ([string]$LocalInventory.total_bytes),
        "--expected-digest", ([string]$LocalInventory.aggregate_sha256)
    )
}

function Invoke-ModalModule {
    param(
        [Parameter(Mandatory)][string]$ModuleName,
        [Parameter(Mandatory)][string]$ModuleArguments
    )

    Write-Stage $ModuleName
    Invoke-CheckedCommand "modal" @(
        "run",
        $modalRunner,
        "--module",
        $ModuleName,
        "--arguments=$ModuleArguments"
    )
}

Push-Location $projectRoot
try {
    Assert-Prerequisites
    if ($IndexStagedOnly) {
        Write-Stage "reuse-staged-artifacts"
        Write-Host "STAGED INPUT: $localProcessed"
    } else {
        $activeRawVideos = Initialize-SliceInput

        Write-Stage "upload-inputs"
        Ensure-ModalVolume
        $localInventory = New-LocalInventory
        if ($SkipUpload) {
            Write-Host "SKIPPED: local raw/caption upload"
        } else {
            $remoteInventory = Get-RemoteInventory
            $uploadPlan = New-UploadPlan -RemoteInventory $remoteInventory
            Assert-ModalUploadApproved -ChangedBytes ([long]$uploadPlan.changed_bytes)
            Invoke-ResumableUpload -UploadPlan $uploadPlan
            if ($DryRun -or (Test-Path -LiteralPath $localCaptions -PathType Container)) {
                Invoke-CheckedCommand "modal" @(
                    "volume", "put", "--force", $VolumeName,
                    $localCaptions, "/captions"
                )
            }
        }
        Assert-RemoteInventory -LocalInventory $localInventory

        $forceArgument = if ($Force) { " --force" } else { "" }
        Invoke-ModalModule "module1" (
        "--input /data/raw_videos --output /data/processed " +
        "--workers 1 --device cuda"
    )
        Invoke-ModalModule "module2" (
        "--metadata-dir /data/processed/metadata " +
        "--keyframe-dir /data/processed/keyframes " +
        "--output-dir /data/processed/embeddings/visual " +
        "--model-id ViT-B-32::openai --device cuda --precision fp16 " +
        "--batch-size 64 --num-workers 4$forceArgument"
    )
        Invoke-ModalModule "module3" (
        "--video-dir /data/raw_videos " +
        "--metadata-dir /data/processed/metadata " +
        "--caption-dir /data/captions --output-dir /data/processed " +
        "--whisper-size medium --llm-provider local " +
        "--llm-model Qwen/Qwen2.5-7B-Instruct --device cuda " +
        "--concurrency 1$forceArgument"
    )
        Invoke-ModalModule "module4" (
        "--keyframe-dir /data/processed/keyframes " +
        "--metadata-dir /data/processed/metadata " +
        "--output-dir /data/processed/ocr --device cuda:0 " +
        "--batch-size 16 --workers 1$forceArgument"
    )
        Invoke-ModalModule "module5" (
        "--keyframe-dir /data/processed/keyframes " +
        "--metadata-dir /data/processed/metadata " +
        "--output-dir /data/processed/object_detection " +
        "--run-yolo-world --yolo-world-model yolov8s-world.pt " +
        "--device cuda --batch-size 16$forceArgument"
    )
        Invoke-ModalModule "module6" (
        "--asr-dir /data/processed/transcripts " +
        "--summary-dir /data/processed/summaries " +
        "--ocr-dir /data/processed/ocr " +
        "--output-dir /data/processed/embeddings " +
        "--model-name dangvantuan/vietnamese-embedding " +
        "--model-revision 4ab46e46ba5902328ba0742e489e75f787932f2b " +
        "--max-length 256 --device cuda --batch-size 128$forceArgument"
    )

        Write-Stage "pull-artifacts"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Force -Path $stagingParent | Out-Null
        }
        Invoke-CheckedCommand "modal" @(
            "volume", "get", "--force", $VolumeName,
            "/processed", $stagingParent
        )
    }

    Write-Stage "docker-indexing"
    if ($SkipIndex) {
        Write-Host "SKIPPED: Docker indexing was disabled with -SkipIndex."
    } else {
        Set-AicComposeDataRoot -Path $stagingParent
        if (-not $DryRun) {
            foreach ($directory in @(
                "etcd", "minio", "milvus", "elasticsearch"
            )) {
                New-Item -ItemType Directory -Force -Path (
                    Join-Path $stagingParent "databases\$directory"
                ) | Out-Null
            }
        }
        Invoke-CheckedCommand "docker" @("compose", "down", "--remove-orphans")
        Invoke-CheckedCommand "docker" @(
            "compose", "up", "-d", "--build",
            "etcd", "minio", "milvus-standalone", "elasticsearch"
        )
        $candidateDockerStarted = -not $DryRun
        Invoke-CheckedCommand "docker" @(
            "compose", "build", "indexing"
        )

        $indexArguments = @(
            "compose", "run", "--rm", "indexing",
            "python", "-m", "src.indexing.cli", "--force"
        )
        if ($ResetIndex) {
            $indexArguments += "--reset-all"
        }
        Invoke-CheckedCommand "docker" $indexArguments

        Write-Stage "verify-and-publish"
        Invoke-CheckedCommand "docker" @(
            "compose", "run", "--rm", "indexing",
            "python", "-m", "src.indexing.publish_cli",
            "--data-dir", "/workspace/data/processed",
            "--dataset-id", $resolvedDatasetId,
            "--manifest-path",
            "/workspace/data/processed/dataset-manifest.json",
            "--building-manifest-path",
            "/workspace/data/processed/dataset-manifest.building.json"
        )
        Invoke-CheckedCommand $pythonExe @(
            "-m", "scripts.btc_storage_manager", "validate-stage",
            "--candidate-root", $stagingParent
        )

        Write-Stage "online-contract-validation"
        $readyFingerprint = "sha256:" + ("0" * 64)
        if (-not $DryRun) {
            $readyManifestPath = Join-Path $localProcessed "dataset-manifest.json"
            $readyManifest = Get-Content -LiteralPath $readyManifestPath -Raw |
                ConvertFrom-Json
            $readyFingerprint = [string]$readyManifest.dataset_fingerprint
            if (
                $readyManifest.status -ne "READY" -or
                $readyFingerprint -notmatch '^sha256:[0-9a-f]{64}$'
            ) {
                throw "Offline publisher did not produce a valid READY manifest."
            }

            $onlineSmoke = @{
                visual_features = @(1.0) + (@(0.0) * 511)
                ocr_features = @(1.0) + (@(0.0) * 767)
                asr_features = @(1.0) + (@(0.0) * 767)
                summary_features = @(1.0) + (@(0.0) * 767)
            }
            $onlineSmokeJson = $onlineSmoke | ConvertTo-Json -Depth 3 -Compress
            $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
            [System.IO.File]::WriteAllText(
                $onlineSmokePath,
                $onlineSmokeJson,
                $utf8WithoutBom
            )
        }
        try {
            Invoke-CheckedCommand "docker" @(
                "compose", "run", "--rm",
                "-e", "AIC_ONLINE_MILVUS_URI=http://milvus-standalone:19530",
                "-e", "AIC_ONLINE_ES_URI=http://elasticsearch:9200",
                "-e", "AIC_ONLINE_SQLITE_PATH=/workspace/data/metadata.db",
                "-e", (
                    "AIC_ONLINE_DATASET_MANIFEST_PATH=" +
                    "/workspace/data/processed/dataset-manifest.json"
                ),
                "-e", "AIC_ONLINE_DATA_ROOT=/workspace/data/processed",
                "-e", "AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT=$readyFingerprint",
                "-e", "AIC_ONLINE_DATASET_MANIFEST_REQUIRED=true",
                "indexing", "python", "-m", "online.validate_contract",
                "--fail-on-partial", "--encoder-smoke-json",
                "/workspace/data/processed/online-encoder-smoke.json"
            )
        }
        finally {
            if (-not $DryRun -and (Test-Path -LiteralPath $onlineSmokePath)) {
                [System.IO.File]::Delete($onlineSmokePath)
            }
        }
        Invoke-CheckedCommand "docker" @("compose", "ps")
        Invoke-CheckedCommand "docker" @("compose", "down", "--remove-orphans")
        $candidateDockerStarted = $false
    }

    Write-Host "`nOffline candidate completed successfully." -ForegroundColor Green
    Write-Host "Staged artifacts: $localProcessed"
    if (-not $SkipIndex) {
        Write-Host "Stop databases later with: docker compose down"
    }
}
finally {
    if ($candidateDockerStarted -and -not $DryRun) {
        & docker compose down --remove-orphans | Out-Host
        $candidateDockerStarted = $false
    }
    [Environment]::SetEnvironmentVariable(
        "AIC_LOCAL_DATA_ROOT", $previousLocalDataRoot, "Process"
    )
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", $previousPythonUtf8, "Process")
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", $previousPythonIoEncoding, "Process")
    [Environment]::SetEnvironmentVariable(
        "AIC_MODAL_DATA_VOLUME", $previousModalDataVolume, "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "MODAL_ENVIRONMENT", $previousModalEnvironment, "Process"
    )
    Pop-Location
}
