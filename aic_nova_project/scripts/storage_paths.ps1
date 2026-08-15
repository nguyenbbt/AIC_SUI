function Get-AicDotEnvValue {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return ""
    }
    $matches = @(
        Get-Content -LiteralPath $Path |
            Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" }
    )
    if ($matches.Count -gt 1) {
        throw "Duplicate $Name entries in $Path"
    }
    if ($matches.Count -eq 0) {
        return ""
    }
    return (($matches[0] -split "=", 2)[1]).Trim().Trim('"').Trim("'")
}

function Resolve-AicLocalDataRoot {
    param(
        [string]$ExplicitStorageRoot = "",
        [Parameter(Mandatory)][string]$ProjectRoot,
        [Parameter(Mandatory)][string]$EnvFile
    )
    $configured = $false
    $candidate = $ExplicitStorageRoot.Trim()
    if (-not [string]::IsNullOrWhiteSpace($candidate)) {
        $configured = $true
    } else {
        $candidate = [Environment]::GetEnvironmentVariable(
            "AIC_LOCAL_DATA_ROOT", "Process"
        )
        if (-not [string]::IsNullOrWhiteSpace($candidate)) {
            $configured = $true
        } else {
            $candidate = Get-AicDotEnvValue `
                -Path $EnvFile `
                -Name "AIC_LOCAL_DATA_ROOT"
            if (-not [string]::IsNullOrWhiteSpace($candidate)) {
                $configured = $true
            } else {
                $candidate = Join-Path $ProjectRoot "data"
            }
        }
    }

    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $ProjectRoot $candidate
    }
    $resolved = [System.IO.Path]::GetFullPath($candidate)
    if ($configured -and -not (Test-Path -LiteralPath $resolved -PathType Container)) {
        throw "Configured AIC_LOCAL_DATA_ROOT is unavailable: $resolved"
    }
    return $resolved
}

function Set-AicComposeDataRoot {
    param([Parameter(Mandatory)][string]$Path)
    [Environment]::SetEnvironmentVariable(
        "AIC_LOCAL_DATA_ROOT",
        $Path.Replace("\", "/"),
        "Process"
    )
}
