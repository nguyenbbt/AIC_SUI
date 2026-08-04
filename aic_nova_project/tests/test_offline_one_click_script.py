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
    ]
    offsets = [output.index(stage) for stage in expected_stages]
    assert offsets == sorted(offsets)
    assert "ViT-B-32::openai" in output
    assert "Qwen/Qwen2.5-1.5B-Instruct" in output
    assert "modal volume get" in output
    assert "python -m src.indexing.cli" in output
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


def test_modal_pull_targets_data_parent_to_avoid_nested_processed_dir():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '$localData = Join-Path $projectRoot "data"' in script
    assert '"/processed", $localData' in script
    assert '"/processed", $localProcessed' not in script


def test_indexing_worker_is_rebuilt_before_it_runs():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    build = '"compose", "build", "indexing"'
    run = '"compose", "run", "--rm", "indexing"'

    assert build in script
    assert run in script
    assert script.index(build) < script.index(run)
