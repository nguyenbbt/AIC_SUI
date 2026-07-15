"""
Tests for the IndexingOrchestrator rollback logic.

Uses mock clients to simulate failures and verify that rollback
functions are called correctly when a downstream DB fails.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from src.indexing.orchestrator import IndexingOrchestrator
from src.indexing.clients.milvus_client import (
    VISUAL_COLLECTION,
    ASR_COLLECTION,
    SUMMARY_COLLECTION,
)
from src.indexing.clients.es_client import OCR_INDEX, ASR_INDEX, SUMMARY_INDEX


@pytest.fixture
def mock_clients():
    """Create mock instances of all 3 clients."""
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    return milvus, es, tabular


@pytest.fixture
def orchestrator(mock_clients):
    """Create an orchestrator with mock clients."""
    milvus, es, tabular = mock_clients
    return IndexingOrchestrator(
        milvus_client=milvus,
        es_client=es,
        tabular_client=tabular,
        batch_size=100,
    )


class TestRollbackOnEsFailure:
    """Test: If Elasticsearch fails after Milvus succeeds, Milvus is rolled back."""

    @patch("src.indexing.orchestrator.load_visual_embeddings", return_value=[
        {"frame_id": "f1", "video_id": "V1", "shot_id": 0, "embedding": [0.1] * 512}
    ])
    @patch("src.indexing.orchestrator.load_text_asr_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_text_summary_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_ocr_texts", return_value=[
        {"frame_id": "f1", "video_id": "V1", "shot_id": "0", "ocr_text_concat": "test"}
    ])
    @patch("src.indexing.orchestrator.load_asr_transcripts", return_value=[])
    @patch("src.indexing.orchestrator.load_video_summary", return_value=[])
    @patch("src.indexing.orchestrator.load_metadata_and_objects", return_value=([], []))
    def test_milvus_rollback_on_es_failure(
        self,
        mock_meta,
        mock_sum_text,
        mock_asr_text,
        mock_ocr,
        mock_sum_emb,
        mock_asr_emb,
        mock_vis,
        orchestrator,
        mock_clients,
    ):
        milvus, es, tabular = mock_clients

        # Make ES bulk_index raise an exception
        es.bulk_index.side_effect = Exception("ES connection timeout")

        result = orchestrator.process_video("V1", Path("/fake"), visual_dim=512, text_dim=768)

        # Should return False (failure)
        assert result is False

        # Milvus insert should have been called (succeeded before ES failed)
        assert milvus.insert_batch.called

        # Milvus rollback (delete_by_video_id) should have been called for all collections
        rollback_calls = milvus.delete_by_video_id.call_args_list
        # Should include the initial cleanup (3 calls) + rollback (3 calls)
        # Initial cleanup: VISUAL, ASR, SUMMARY collections
        # Rollback: VISUAL, ASR, SUMMARY collections
        assert len(rollback_calls) >= 6


class TestRollbackOnSqliteFailure:
    """Test: If SQLite fails after both Milvus and ES succeed, both are rolled back."""

    @patch("src.indexing.orchestrator.load_visual_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_text_asr_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_text_summary_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_ocr_texts", return_value=[])
    @patch("src.indexing.orchestrator.load_asr_transcripts", return_value=[])
    @patch("src.indexing.orchestrator.load_video_summary", return_value=[])
    @patch("src.indexing.orchestrator.load_metadata_and_objects", return_value=(
        [{"frame_id": "f1", "video_id": "V1", "shot_id": 0, "timestamp": 1.0}],
        [{"frame_id": "f1", "label": "person", "confidence": 0.9,
          "x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100, "model_source": "yolo"}],
    ))
    def test_both_rollback_on_sqlite_failure(
        self,
        mock_meta,
        mock_sum_text,
        mock_asr_text,
        mock_ocr,
        mock_sum_emb,
        mock_asr_emb,
        mock_vis,
        orchestrator,
        mock_clients,
    ):
        milvus, es, tabular = mock_clients

        # Make SQLite insert_metadata_batch raise an exception
        tabular.insert_metadata_batch.side_effect = Exception("SQLite disk full")

        result = orchestrator.process_video("V1", Path("/fake"), visual_dim=512, text_dim=768)

        # Should return False (failure)
        assert result is False

        # Both Milvus and ES rollback should have been called
        milvus_delete_calls = [
            c for c in milvus.delete_by_video_id.call_args_list
        ]
        es_delete_calls = [
            c for c in es.delete_by_video_id.call_args_list
        ]
        # More than the initial cleanup calls = rollback happened
        assert len(milvus_delete_calls) >= 6  # 3 cleanup + 3 rollback
        assert len(es_delete_calls) >= 6  # 3 cleanup + 3 rollback


class TestSuccessfulProcessing:
    """Test: All inserts succeed → process_video returns True."""

    @patch("src.indexing.orchestrator.load_visual_embeddings", return_value=[
        {"frame_id": "f1", "video_id": "V1", "shot_id": 0, "embedding": [0.1] * 512}
    ])
    @patch("src.indexing.orchestrator.load_text_asr_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_text_summary_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_ocr_texts", return_value=[
        {"frame_id": "f1", "video_id": "V1", "shot_id": "0", "ocr_text_concat": "xin chào"}
    ])
    @patch("src.indexing.orchestrator.load_asr_transcripts", return_value=[])
    @patch("src.indexing.orchestrator.load_video_summary", return_value=[])
    @patch("src.indexing.orchestrator.load_metadata_and_objects", return_value=(
        [{"frame_id": "f1", "video_id": "V1", "shot_id": 0, "timestamp": 1.0}],
        [],
    ))
    def test_success_returns_true(
        self,
        mock_meta,
        mock_sum_text,
        mock_asr_text,
        mock_ocr,
        mock_sum_emb,
        mock_asr_emb,
        mock_vis,
        orchestrator,
        mock_clients,
    ):
        milvus, es, tabular = mock_clients

        result = orchestrator.process_video("V1", Path("/fake"), visual_dim=512, text_dim=768)

        assert result is True
        assert milvus.insert_batch.called
        assert es.bulk_index.called
        assert tabular.insert_metadata_batch.called


class TestGracefulDegradation:
    """Test: Missing data streams don't crash the pipeline."""

    @patch("src.indexing.orchestrator.load_visual_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_text_asr_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_text_summary_embeddings", return_value=[])
    @patch("src.indexing.orchestrator.load_ocr_texts", return_value=[])
    @patch("src.indexing.orchestrator.load_asr_transcripts", return_value=[])
    @patch("src.indexing.orchestrator.load_video_summary", return_value=[])
    @patch("src.indexing.orchestrator.load_metadata_and_objects", return_value=([], []))
    def test_empty_data_still_succeeds(
        self,
        mock_meta,
        mock_sum_text,
        mock_asr_text,
        mock_ocr,
        mock_sum_emb,
        mock_asr_emb,
        mock_vis,
        orchestrator,
        mock_clients,
    ):
        milvus, es, tabular = mock_clients

        result = orchestrator.process_video("V1", Path("/fake"), visual_dim=512, text_dim=768)

        # Should succeed even with zero data — graceful degradation
        assert result is True
