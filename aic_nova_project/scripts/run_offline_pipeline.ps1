[CmdletBinding()]
param(
    [string]$VolumeName = "aic-nova-offline-data",
    [switch]$Force,
    [switch]$ResetIndex,
    [switch]$SkipIndex,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modalRunner = Join-Path $projectRoot "scripts\offline_modal_runner.py"
$localRawVideos = Join-Path $projectRoot "data\raw_videos"
$localCaptions = Join-Path $projectRoot "data\captions"
$localProcessed = Join-Path $projectRoot "data\processed"

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

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $displayCommand"
    }
}

function Assert-Prerequisites {
    if ($DryRun) {
        return
    }
    foreach ($commandName in @("modal", "docker")) {
        if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
            throw "Required command is not available: $commandName"
        }
    }
    if (-not (Test-Path -LiteralPath $modalRunner -PathType Leaf)) {
        throw "Modal runner not found: $modalRunner"
    }

    $videoFiles = @(
        Get-ChildItem -LiteralPath $localRawVideos -File -Recurse |
            Where-Object { $_.Extension -in @(".mp4", ".mkv", ".avi", ".webm") }
    )
    if ($videoFiles.Count -eq 0) {
        throw "No input videos found in $localRawVideos"
    }
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
    $volumes = @($volumeJson | ConvertFrom-Json)
    if ($VolumeName -notin $volumes.name) {
        Invoke-CheckedCommand "modal" @("volume", "create", $VolumeName)
    } else {
        Write-Host "Modal volume already exists: $VolumeName"
    }
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

    Write-Stage "upload-inputs"
    Ensure-ModalVolume
    Invoke-CheckedCommand "modal" @(
        "volume", "put", "--force", $VolumeName, $localRawVideos, "/"
    )
    if ($DryRun -or (Test-Path -LiteralPath $localCaptions -PathType Container)) {
        Invoke-CheckedCommand "modal" @(
            "volume", "put", "--force", $VolumeName, $localCaptions, "/"
        )
    }

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
        "--llm-model Qwen/Qwen2.5-1.5B-Instruct --device cuda " +
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
        "--device cuda --batch-size 128$forceArgument"
    )

    Write-Stage "pull-artifacts"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $localProcessed | Out-Null
    }
    Invoke-CheckedCommand "modal" @(
        "volume", "get", "--force", $VolumeName,
        "/processed", $localProcessed
    )

    Write-Stage "docker-indexing"
    if ($SkipIndex) {
        Write-Host "SKIPPED: Docker indexing was disabled with -SkipIndex."
    } else {
        $envFile = Join-Path $projectRoot ".env"
        if (-not $DryRun -and -not (Test-Path -LiteralPath $envFile)) {
            Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") `
                -Destination $envFile
        }
        Invoke-CheckedCommand "docker" @(
            "compose", "up", "-d", "--build",
            "etcd", "minio", "milvus-standalone", "elasticsearch"
        )

        $indexArguments = @(
            "compose", "run", "--rm", "indexing",
            "python", "-m", "src.indexing.cli", "--force"
        )
        if ($ResetIndex) {
            $indexArguments += "--reset-all"
        }
        Invoke-CheckedCommand "docker" $indexArguments
        Invoke-CheckedCommand "docker" @("compose", "ps")
    }

    Write-Host "`nOffline pipeline completed successfully." -ForegroundColor Green
    Write-Host "Local artifacts: $localProcessed"
    if (-not $SkipIndex) {
        Write-Host "Stop databases later with: docker compose down"
    }
}
finally {
    Pop-Location
}
