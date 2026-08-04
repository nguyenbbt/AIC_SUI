from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.indexing.orchestrator import IndexingOrchestrator


def test_run_raises_when_any_video_fails(tmp_path: Path):
    orchestrator = IndexingOrchestrator(
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    orchestrator.process_video = MagicMock(return_value=False)

    with patch(
        "src.indexing.orchestrator.discover_video_ids",
        return_value=["V001"],
    ), patch(
        "src.indexing.orchestrator.detect_embedding_dim",
        return_value=2,
    ):
        with pytest.raises(RuntimeError, match="V001"):
            orchestrator.run(tmp_path)
