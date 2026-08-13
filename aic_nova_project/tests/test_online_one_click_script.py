from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_online_stack.ps1"


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


def test_start_dry_run_orders_the_full_online_stack():
    result = _run_dry_run("-Action", "Start")

    assert result.returncode == 0, result.stderr
    expected_stages = [
        "STAGE: prerequisites",
        "STAGE: docker-infrastructure",
        "STAGE: modal-encoder-deploy",
        "STAGE: qwen-vlm",
        "STAGE: contract-validation",
        "STAGE: api",
        "STAGE: ui",
        "STAGE: readiness",
    ]
    offsets = [result.stdout.index(stage) for stage in expected_stages]
    assert offsets == sorted(offsets)
    assert "AIC_ONLINE_TRAKE_ENABLED=true" in result.stdout
    assert "AIC_ONLINE_VQA_ENABLED=true" in result.stdout
    assert "http://127.0.0.1:5173" in result.stdout


def test_start_uses_only_existing_indexes_and_full_contract_validation():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"etcd", "minio", "milvus-standalone", "elasticsearch"' in script
    assert "generate_online_modal_smoke_vectors" in script
    assert "online.validate_contract" in script
    assert '"--fail-on-partial"' in script
    assert "indexing" not in script.lower()
    assert "--reset-all" not in script
    assert "down -v" not in script.lower()
    assert '"compose", "down"' not in script


def test_vqa_fails_closed_or_starts_an_explicitly_pinned_image():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"http://127.0.0.1:8001/v1"' in script
    assert '$($VqaBaseUrl.TrimEnd(\'/\'))/models' in script
    assert "QwenVllmImage" in script
    assert "PINNED vLLM image" in script
    assert '"--gpus", "all"' in script
    assert "Qwen/Qwen3.5-4B" in script
    assert "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in script
    assert "vllm/vllm-openai:latest" not in script

    latest = _run_dry_run(
        "-Action",
        "Start",
        "-QwenVllmImage",
        "vllm/vllm-openai:latest",
    )
    assert latest.returncode != 0
    assert "PINNED vLLM image" in latest.stderr


def test_without_vqa_keeps_kis_and_trake_available():
    result = _run_dry_run("-Action", "Start", "-WithoutVQA")

    assert result.returncode == 0, result.stderr
    assert "STAGE: qwen-vlm" not in result.stdout
    assert "AIC_ONLINE_TRAKE_ENABLED=true" in result.stdout
    assert "AIC_ONLINE_VQA_ENABLED=false" in result.stdout


def test_stop_is_scoped_to_owned_processes_and_preserves_volumes():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Get-CimInstance Win32_Process" in script
    assert "retrieval_api.main:app" in script
    assert "vite" in script
    assert '"compose", "stop", "elasticsearch", "milvus-standalone", "minio", "etcd"' in script
    assert "Stop-Process" in script
    assert "processes.json" in script


def test_runner_supports_start_stop_and_status_actions():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '[ValidateSet("Start", "Stop", "Status")]' in script
    for action in ("Start", "Stop", "Status"):
        result = _run_dry_run("-Action", action)
        assert result.returncode == 0, result.stderr
