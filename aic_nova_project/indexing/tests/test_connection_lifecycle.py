from unittest.mock import MagicMock

import pytest

from src.indexing.orchestrator import IndexingOrchestrator


def test_run_disconnects_every_attempted_client_on_failure(tmp_path):
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    orchestrator = IndexingOrchestrator(milvus, es, tabular)
    orchestrator._run_connected = MagicMock(
        side_effect=RuntimeError("schema audit failed")
    )

    with pytest.raises(RuntimeError, match="schema audit failed"):
        orchestrator.run(tmp_path)

    tabular.disconnect.assert_called_once_with()
    es.disconnect.assert_called_once_with()
    milvus.disconnect.assert_called_once_with()
