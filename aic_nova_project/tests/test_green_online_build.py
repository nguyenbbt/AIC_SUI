from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_green_indexing.ps1"
PREVIEW_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_green_preview.ps1"


def test_green_runner_never_targets_blue_compose_or_ui_ports() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'docker-compose.green.yml' in script
    assert '"-p", $composeProject' in script
    assert '$composeProject = "aic-nova-green"' in script
    assert 'docker-compose.yml' not in script.replace(
        'docker-compose.green.yml', ''
    )
    assert 'run_online_stack.ps1' not in script
    assert '5173' not in script
    assert '8000' not in script
    assert '"compose", "down"' not in script


def test_green_runner_uses_fresh_bulk_rebuild_and_ready_gates() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"--force", "--reset-all", "--bulk-rebuild"' in script
    assert 'src.indexing.publish_cli' in script
    assert 'online.validate_contract' in script
    assert '"--fail-on-partial"' in script
    assert 'dataset-manifest.building.json' in script
    assert 'dataset-manifest.json' in script


def test_green_runner_protects_active_blue_mounts_and_source_artifacts() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'Assert-GreenRootDoesNotOverlapBlue' in script
    assert '"inspect", "aic_nova_milvus"' in script
    assert 'AIC_GREEN_BACKUP_ROOT' in script
    assert 'read-only' in script.lower()
    assert 'Remove-Item -LiteralPath $candidateRoot' not in script
    assert 'Copy-Item' not in script


def test_green_build_starts_preview_only_after_ready_validation() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    validation = script.index('Write-Stage "green-full-contract-validation"')
    preview = script.index('Invoke-CheckedCommand $previewScript')
    assert validation < preview
    assert '[switch]$SkipPreview' in script
    assert '"-Action", "Start"' in script


def test_green_preview_uses_separate_ports_and_requires_ready_manifest() -> None:
    script = PREVIEW_SCRIPT_PATH.read_text(encoding="utf-8")
    vite_config = (
        PROJECT_ROOT / "ui" / "vite.green.config.js"
    ).read_text(encoding="utf-8")

    assert '[ValidateSet("Start", "Stop", "Status")]' in script
    assert 'dataset-manifest.json' in script
    assert 'status -ne "READY"' in script
    assert 'AIC_ONLINE_DATASET_MANIFEST_REQUIRED = "true"' in script
    assert 'http://127.0.0.1:19531' in script
    assert 'http://127.0.0.1:19200' in script
    assert '"--port", "8001"' in script
    assert '"--port", "5174"' in script
    assert 'http://127.0.0.1:8001/health/ready' in script
    assert 'http://127.0.0.1:5174/api/health/ready' in script
    assert 'target: "http://127.0.0.1:8001"' in vite_config
    assert 'port: 5174' in vite_config
    assert 'run_online_stack.ps1' not in script
