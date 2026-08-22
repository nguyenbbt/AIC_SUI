[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BackupRoot,
    [Parameter(Mandatory)][string]$VideosRoot,
    [ValidateRange(1, 1000000)][int]$ExpectedVideoCount = 873,
    [ValidateRange(1, 100000000)][int]$ExpectedKeyframeCount = 293427,
    [switch]$RequireObjects,
    [switch]$UpdateEnv,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. "$(Join-Path $PSScriptRoot 'storage_paths.ps1')"
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Project Python executable is unavailable: $pythonExe"
}
$resolvedBackupRoot = (Resolve-Path -LiteralPath $BackupRoot).Path
$resolvedVideosRoot = (Resolve-Path -LiteralPath $VideosRoot).Path

$auditArguments = @(
    "-m", "scripts.btc_storage_manager", "audit-backup",
    "--backup-root", $resolvedBackupRoot,
    "--videos-root", $resolvedVideosRoot,
    "--expected-videos", [string]$ExpectedVideoCount,
    "--expected-keyframes", [string]$ExpectedKeyframeCount
)
if ($RequireObjects) {
    $auditArguments += "--require-objects"
}

Write-Host "STAGE: audit-backup" -ForegroundColor Cyan
$auditOutput = @(& $pythonExe @auditArguments)
if ($LASTEXITCODE -ne 0) {
    throw "Backup artifact audit failed with exit code $LASTEXITCODE."
}
$audit = ($auditOutput -join "`n") | ConvertFrom-Json

Write-Host "STAGE: materialize-processed-view" -ForegroundColor Cyan
$createdLinks = @()
foreach ($property in $audit.links.PSObject.Properties) {
    $relativePath = [string]$property.Name
    $source = [System.IO.Path]::GetFullPath([string]$property.Value)
    $target = Join-Path $resolvedBackupRoot ($relativePath.Replace("/", "\"))
    $target = [System.IO.Path]::GetFullPath($target)
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Junction source is unavailable: $source"
    }

    if (Test-Path -LiteralPath $target) {
        $existing = Get-Item -LiteralPath $target -Force
        $existingTarget = @($existing.Target) | Select-Object -First 1
        if (
            $existing.LinkType -ne "Junction" -or
            [string]::IsNullOrWhiteSpace([string]$existingTarget) -or
            [System.IO.Path]::GetFullPath([string]$existingTarget) -ne $source
        ) {
            throw (
                "Refusing to overwrite existing non-matching path: $target. " +
                "Expected junction target: $source"
            )
        }
        Write-Host "PASS existing junction: $relativePath -> $source"
        continue
    }

    Write-Host "CREATE junction: $relativePath -> $source"
    if (-not $DryRun) {
        [System.IO.Directory]::CreateDirectory(
            (Split-Path -Parent $target)
        ) | Out-Null
        New-Item -ItemType Junction -Path $target -Target $source | Out-Null
        $createdLinks += $relativePath
    }
}

# processed/object_detection is intentionally absent until the complete M5
# artifact set passes the same audit with -RequireObjects.
if (-not $audit.ready_for_indexing) {
    Write-Host (
        "M5 GATE: WAITING; Object Detection is not yet complete. " +
        "Do not run M7 or Online."
    ) -ForegroundColor Yellow
} else {
    Write-Host "M5 GATE: PASS; layout is ready for M7." -ForegroundColor Green
}

if ($UpdateEnv) {
    Write-Host "STAGE: configure-storage-root" -ForegroundColor Cyan
    Write-Host "SET AIC_LOCAL_DATA_ROOT=$resolvedBackupRoot"
    if (-not $DryRun) {
        Set-AicDotEnvValue `
            -Path $envFile `
            -Name "AIC_LOCAL_DATA_ROOT" `
            -Value $resolvedBackupRoot
    }
}

if (-not $DryRun) {
    $journalRoot = Join-Path $resolvedBackupRoot ".migration"
    [System.IO.Directory]::CreateDirectory($journalRoot) | Out-Null
    $journal = [ordered]@{
        schema_version = 1
        created_at = [DateTimeOffset]::UtcNow.ToString("o")
        backup_root = $resolvedBackupRoot
        videos_root = $resolvedVideosRoot
        ready_for_indexing = [bool]$audit.ready_for_indexing
        created_links = $createdLinks
        artifact_counts = $audit.artifact_counts
    }
    $journalPath = Join-Path $journalRoot "backup-online-layout.json"
    $journalJson = $journal | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $journalPath,
        $journalJson,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "JOURNAL: $journalPath"
}

Write-Host "Backup Online layout preparation complete." -ForegroundColor Green
