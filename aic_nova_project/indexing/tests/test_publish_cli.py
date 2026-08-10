import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing import publish_cli


@pytest.fixture
def cli_dependencies(monkeypatch):
    snapshot = object()
    draft = MagicMock(dataset_fingerprint=f"sha256:{'a' * 64}")
    ready = MagicMock(
        dataset_id="run-001",
        dataset_fingerprint=f"sha256:{'a' * 64}",
    )
    monkeypatch.setattr(
        publish_cli,
        "collect_full_snapshot",
        lambda **kwargs: snapshot,
    )
    monkeypatch.setattr(
        publish_cli,
        "record_counts",
        lambda value: {"videos": 2},
    )
    monkeypatch.setattr(
        publish_cli,
        "build_manifest_draft",
        lambda **kwargs: draft,
    )
    write_draft = MagicMock()
    publish_ready = MagicMock(return_value=ready)
    monkeypatch.setattr(publish_cli, "write_manifest_draft", write_draft)
    monkeypatch.setattr(publish_cli, "publish_ready_manifest", publish_ready)
    return snapshot, draft, write_draft, publish_ready


def _argv(tmp_path: Path) -> list[str]:
    return [
        "publish-offline-dataset",
        "--data-dir",
        str(tmp_path),
        "--dataset-id",
        "run-001",
        "--manifest-path",
        str(tmp_path / "dataset-manifest.json"),
        "--building-manifest-path",
        str(tmp_path / "dataset-manifest.building.json"),
    ]


def test_publish_cli_does_not_publish_ready_when_verifier_fails(
    tmp_path,
    monkeypatch,
    cli_dependencies,
):
    _, _, write_draft, publish_ready = cli_dependencies
    monkeypatch.setattr(
        publish_cli,
        "build_full_contract_report",
        lambda *args, **kwargs: ["bad vector"],
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert publish_cli.main() == 1
    write_draft.assert_called_once()
    publish_ready.assert_not_called()


def test_publish_cli_publishes_only_after_clean_verification(
    tmp_path,
    monkeypatch,
    cli_dependencies,
):
    _, draft, write_draft, publish_ready = cli_dependencies
    monkeypatch.setattr(
        publish_cli,
        "build_full_contract_report",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert publish_cli.main() == 0
    write_draft.assert_called_once()
    publish_ready.assert_called_once_with(
        draft,
        tmp_path / "dataset-manifest.json",
        verification_errors=[],
    )
