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
        "STAGE: mgpux-qwen-deploy",
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

    assert "mgpux_qwen_vlm.py" in script
    assert "m-gpux.exe" in script
    assert '"app", "stop", $mgpuxAppName, "--yes"' in script
    assert "AIC_ONLINE_QWEN_VLM_API_KEY" in script
    assert 'AIC_ONLINE_VQA_TOTAL_TIMEOUT_SEC = "180"' in script
    assert 'AIC_ONLINE_VQA_VLM_TIMEOUT_SEC = "120"' in script
    assert "Qwen/Qwen3.5-4B" in script
    assert "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in script
    assert "QwenVllmImage" not in script
    assert '"--gpus", "all"' not in script
    assert "vllm/vllm-openai:latest" not in script

    dry_run = _run_dry_run("-Action", "Start")
    assert dry_run.returncode == 0, dry_run.stderr
    assert "STAGE: mgpux-qwen-deploy" in dry_run.stdout
    assert "modal.exe deploy" in dry_run.stdout
    assert "AIC_ONLINE_QWEN_VLM_API_KEY=<redacted>" in dry_run.stdout


def test_without_vqa_keeps_kis_and_trake_available():
    result = _run_dry_run("-Action", "Start", "-WithoutVQA")

    assert result.returncode == 0, result.stderr
    assert "STAGE: mgpux-qwen-deploy" not in result.stdout
    assert "AIC_ONLINE_TRAKE_ENABLED=true" in result.stdout
    assert "AIC_ONLINE_VQA_ENABLED=false" in result.stdout


def test_stop_is_scoped_to_owned_processes_and_preserves_volumes():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Get-CimInstance Win32_Process" in script
    assert "retrieval_api.main:app" in script
    assert "vite" in script
    assert '"compose", "stop", "elasticsearch", "milvus-standalone", "minio", "etcd"' in script
    assert '"app", "stop", $mgpuxAppName, "--yes"' in script
    assert "Stop-Process" in script
    assert "processes.json" in script
    assert "Startup failed; stopping partial API/UI processes." in script
    assert "taskkill.exe" in script
    assert "Stop-PartialApiAndUi -Processes $processes" in script


def test_mgpux_qwen_service_is_reproducible_and_scales_to_zero():
    service_path = PROJECT_ROOT / "scripts" / "mgpux_qwen_vlm.py"
    service = service_path.read_text(encoding="utf-8")

    assert 'modal.App("m-gpux-llm-api")' in service
    assert '"Qwen/Qwen3.5-4B"' in service
    assert '"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"' in service
    assert '"vllm==0.25.1"' in service
    assert 'gpu="L4"' in service
    assert "min_containers=0" in service
    assert "max_containers=1" in service
    assert '"m-gpux-hf-cache"' in service
    assert '"m-gpux-vllm-cache"' in service
    assert '"--limit-mm-per-prompt"' in service
    assert '"16384"' in service
    assert '"--api-key"' not in service
    assert 'INTERNAL_VLLM_PORT = "8001"' in service
    assert '"mgpux_qwen_proxy:app"' in service
    assert '"HF_XET_HIGH_PERFORMANCE": "1"' in service
    assert '"--enable-prefix-caching"' not in service


def test_runner_supports_start_stop_and_status_actions():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '[ValidateSet("Start", "Stop", "Status")]' in script
    assert "& $FilePath @Arguments | Out-Host" in script
    for action in ("Start", "Stop", "Status"):
        result = _run_dry_run("-Action", action)
        assert result.returncode == 0, result.stderr


def test_online_runner_uses_shared_explicit_storage_root():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '[string]$StorageRoot = ""' in script
    assert '. "$(Join-Path $PSScriptRoot \'storage_paths.ps1\')"' in script
    assert '$dataRoot = Join-Path $localDataRoot "processed"' in script
    assert '$sqlitePath = Join-Path $localDataRoot "metadata.db"' in script
    assert 'AIC_LOCAL_DATA_ROOT = $localDataRoot' in script


def test_explicit_storage_root_with_spaces_controls_online_paths(tmp_path):
    storage_root = tmp_path / "DATA AIC"
    storage_root.mkdir()

    result = _run_dry_run("-Action", "Start", "-StorageRoot", str(storage_root))

    assert result.returncode == 0, result.stderr
    assert f"ENV: AIC_LOCAL_DATA_ROOT={storage_root}" in result.stdout
    assert f"ENV: AIC_ONLINE_SQLITE_PATH={storage_root / 'metadata.db'}" in result.stdout
    assert f"ENV: AIC_ONLINE_DATA_ROOT={storage_root / 'processed'}" in result.stdout


def test_explicit_missing_storage_root_fails_closed(tmp_path):
    missing = tmp_path / "missing-ssd"

    result = _run_dry_run("-Action", "Start", "-StorageRoot", str(missing))

    assert result.returncode != 0
    assert "Configured AIC_LOCAL_DATA_ROOT is unavailable" in result.stderr
