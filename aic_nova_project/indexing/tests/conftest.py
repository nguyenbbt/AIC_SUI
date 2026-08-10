"""Shared contract fixtures for indexing tests."""

import pytest

import src.indexing.orchestrator as orchestrator_module
from tests.contract_fixtures import canonical_video_record


@pytest.fixture(autouse=True)
def canonical_video_loader(monkeypatch):
    """Keep orchestrator unit tests aligned with the V2 videos contract."""
    monkeypatch.setattr(
        orchestrator_module,
        "load_video_metadata",
        lambda data_dir, video_id: canonical_video_record(video_id),
    )
