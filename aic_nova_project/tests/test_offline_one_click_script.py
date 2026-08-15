from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_offline_pipeline.ps1"


def _run_dry_run(*arguments: str) -> subprocess.CompletedProcess[str]:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required to validate the Windows runner")

    return subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_PATH),
            "-DryRun",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_one_click_dry_run_contains_ordered_offline_stages():
    result = _run_dry_run()

    assert result.returncode == 0, result.stderr
    output = result.stdout
    expected_stages = [
        "STAGE: upload-inputs",
        "STAGE: module1",
        "STAGE: module2",
        "STAGE: module3",
        "STAGE: module4",
        "STAGE: module5",
        "STAGE: module6",
        "STAGE: pull-artifacts",
        "STAGE: docker-indexing",
        "STAGE: verify-and-publish",
        "STAGE: online-contract-validation",
    ]
    offsets = [output.index(stage) for stage in expected_stages]
    assert offsets == sorted(offsets)
    assert "ViT-B-32::openai" in output
    assert "dangvantuan/vietnamese-embedding" in output
    assert "4ab46e46ba5902328ba0742e489e75f787932f2b" in output
    assert "--max-length 256" in output
    assert "Qwen/Qwen2.5-7B-Instruct" in output
    assert "modal volume get" in output
    assert "python -m src.indexing.cli" in output
    assert "python -m src.indexing.publish_cli" in output
    assert "python -m online.validate_contract" in output
    assert "--fail-on-partial" in output
    assert "AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT=" in output
    assert (
        "AIC_ONLINE_DATASET_MANIFEST_PATH="
        "/workspace/data/processed/dataset-manifest.json"
    ) in output
    assert "AIC_ONLINE_DATA_ROOT=/workspace/data/processed" in output
    assert "dataset-manifest.json" in output
    indexing_command = next(
        line
        for line in output.splitlines()
        if "python -m src.indexing.cli" in line
    )
    assert "--reset-all" not in indexing_command


def test_force_and_reset_flags_are_explicitly_forwarded():
    result = _run_dry_run("-Force", "-ResetIndex")

    assert result.returncode == 0, result.stderr
    output = result.stdout
    module2_command = next(
        line
        for line in output.splitlines()
        if "--module module2" in line
    )
    indexing_command = next(
        line
        for line in output.splitlines()
        if "python -m src.indexing.cli" in line
    )
    assert module2_command.endswith("--force")
    assert indexing_command.endswith("--force --reset-all")


def test_modal_volume_json_array_is_flattened_before_name_lookup():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "$volumes = @($volumeJson | ConvertFrom-Json)" not in script
    assert "$volumeNames = @($volumes | ForEach-Object { $_.name })" in script
    assert "$VolumeName -notin $volumeNames" in script


def test_modal_pull_targets_ssd_staging_to_avoid_serving_partial_data():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '[string]$StorageRoot = ""' in script
    assert '. "$(Join-Path $PSScriptRoot \'storage_paths.ps1\')"' in script
    assert 'Join-Path $localData ".staging"' in script
    assert '"/processed", $stagingParent' in script
    assert '"/processed", $localData' not in script
    assert '"/processed", $localProcessed' not in script


def test_offline_runner_has_paid_upload_and_resume_gates():
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    compact = " ".join(script.split())

    assert "[switch]$SkipUpload" in script
    assert "[switch]$ForceUpload" in script
    assert "[switch]$ApprovePaidUpload" in script
    assert '"aic-nova-btc-data"' in script
    assert "Assert-ModalUploadApproved" in script
    assert "Assert-RemoteInventory" in script
    assert (
        'SetEnvironmentVariable( "AIC_MODAL_DATA_VOLUME", $VolumeName, "Process" )'
        in compact
    )
    assert (
        'SetEnvironmentVariable( "AIC_MODAL_DATA_VOLUME", '
        '$previousModalDataVolume, "Process" )'
        in compact
    )


def test_vertical_slice_uses_ssd_hardlink_input_and_exact_video_id():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '[string]$SliceVideoId = ""' in script
    assert "Initialize-SliceInput" in script
    assert "-ItemType HardLink" in script
    assert 'Join-Path $localData ".staging\\inputs"' in script
    assert "$activeRawVideos" in script


def test_full_run_enforces_free_space_and_modal_context_gates():
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    compact = " ".join(script.split())

    assert '[string]$ModalEnvironment = ""' in script
    assert "251.85" in script
    assert "System.IO.DriveInfo" in script
    assert 'Invoke-CheckedCommand "modal" @("profile", "current")' in script
    assert 'SetEnvironmentVariable( "MODAL_ENVIRONMENT"' in compact


def test_missing_remote_inventory_is_treated_as_first_upload_state():
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    function_body = script.split("function Get-RemoteInventory", 1)[1].split(
        "function New-UploadPlan", 1
    )[0]

    assert '$ErrorActionPreference = "Continue"' in function_body
    assert "$remoteGetExitCode = $LASTEXITCODE" in function_body
    assert "$ErrorActionPreference = $previousErrorActionPreference" in function_body
    assert "$remoteGetExitCode -ne 0" in function_body


def test_checked_command_does_not_leak_stdout_into_function_return_values():
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    function_body = script.split("function Invoke-CheckedCommand", 1)[1].split(
        "function Assert-Prerequisites", 1
    )[0]

    assert "& $FilePath @Arguments | Out-Host" in function_body


def test_index_staged_only_skips_all_modal_work_and_resumes_docker():
    result = _run_dry_run(
        "-IndexStagedOnly",
        "-DatasetId",
        "btc-slice-L21-V001",
    )

    assert result.returncode == 0, result.stderr
    assert "STAGE: reuse-staged-artifacts" in result.stdout
    assert "STAGE: module1" not in result.stdout
    assert "modal run" not in result.stdout
    assert "STAGE: docker-indexing" in result.stdout


def test_indexing_worker_is_rebuilt_before_it_runs():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    build = '"compose", "build", "indexing"'
    run = '"compose", "run", "--rm", "indexing"'

    assert build in script
    assert run in script
    assert script.index(build) < script.index(run)


def test_online_validation_uses_ephemeral_smoke_artifact():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"online-encoder-smoke.json"' in script
    assert "[System.IO.File]::WriteAllText" in script
    assert "[System.IO.File]::Delete($onlineSmokePath)" in script


def test_modal_cli_runs_with_utf8_process_encoding():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'SetEnvironmentVariable("PYTHONUTF8", "1", "Process")' in script
    assert (
        'SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")'
        in script
    )
    assert (
        'SetEnvironmentVariable("PYTHONUTF8", $previousPythonUtf8, "Process")'
        in script
    )
    assert (
        'SetEnvironmentVariable('
        '"PYTHONIOENCODING", $previousPythonIoEncoding, "Process")'
        in script
    )
