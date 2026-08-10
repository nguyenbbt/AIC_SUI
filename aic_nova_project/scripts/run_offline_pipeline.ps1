[CmdletBinding()]
param(
    [string]$VolumeName = "aic-nova-offline-data",
    [switch]$Force,
    [switch]$ResetIndex,
    [switch]$SkipIndex,
    [switch]$DryRun,
    [string]$DatasetId = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modalRunner = Join-Path $projectRoot "scripts\offline_modal_runner.py"
$localData = Join-Path $projectRoot "data"
$localRawVideos = Join-Path $projectRoot "data\raw_videos"
$localCaptions = Join-Path $projectRoot "data\captions"
$localProcessed = Join-Path $localData "processed"
$onlineSmokePath = Join-Path $localProcessed "online-encoder-smoke.json"
$previousPythonUtf8 = [Environment]::GetEnvironmentVariable(
    "PYTHONUTF8", "Process"
)
$previousPythonIoEncoding = [Environment]::GetEnvironmentVariable(
    "PYTHONIOENCODING", "Process"
)
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")

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
    $volumes = $volumeJson | ConvertFrom-Json
    $volumeNames = @($volumes | ForEach-Object { $_.name })
    if ($VolumeName -notin $volumeNames) {
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
    $resolvedDatasetId = if ([string]::IsNullOrWhiteSpace($DatasetId)) {
        "aic2026-team-run-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))"
    } else {
        $DatasetId
    }
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
        New-Item -ItemType Directory -Force -Path $localData | Out-Null
    }
    Invoke-CheckedCommand "modal" @(
        "volume", "get", "--force", $VolumeName,
        "/processed", $localData
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
    }

    Write-Host "`nOffline pipeline completed successfully." -ForegroundColor Green
    Write-Host "Local artifacts: $localProcessed"
    if (-not $SkipIndex) {
        Write-Host "Stop databases later with: docker compose down"
    }
}
finally {
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", $previousPythonUtf8, "Process")
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", $previousPythonIoEncoding, "Process")
    Pop-Location
}
