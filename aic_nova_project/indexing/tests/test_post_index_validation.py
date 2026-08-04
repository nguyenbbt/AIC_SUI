from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.indexing.orchestrator as orchestrator_module
from src.indexing.orchestrator import IndexingOrchestrator, VideoSnapshot


def test_post_index_mismatch_identifies_the_failed_stream():
    visual = [
        {
            "frame_id": "V001_00000_050",
            "video_id": "V001",
            "shot_id": 0,
            "embedding": [1.0, 0.0],
        }
    ]
    snapshot = VideoSnapshot(
        milvus={
            orchestrator_module.VISUAL_COLLECTION: [],
            orchestrator_module.ASR_COLLECTION: [],
            orchestrator_module.SUMMARY_COLLECTION: [],
            orchestrator_module.OCR_COLLECTION: [],
        },
        elasticsearch={
            orchestrator_module.OCR_INDEX: [],
            orchestrator_module.ASR_INDEX: [],
            orchestrator_module.SUMMARY_INDEX: [],
        },
        metadata=[],
        objects=[],
    )
    orchestrator = IndexingOrchestrator(
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match=r"milvus\.visual_features: expected=1 actual=0",
    ):
        orchestrator._validate_post_index(
            snapshot,
            visual_records=visual,
            asr_emb_records=[],
            summary_emb_records=[],
            ocr_emb_records=[],
            ocr_text_records=[],
            asr_text_records=[],
            summary_text_records=[],
            metadata_records=[],
            object_records=[],
            visual_dim=2,
            text_dim=2,
            ocr_dim=2,
        )


def test_missing_post_index_record_fails_and_rolls_back(
    monkeypatch,
):
    visual = [
        {
            "frame_id": "V001_00000_050",
            "video_id": "V001",
            "shot_id": 0,
            "embedding": [1.0, 0.0],
        }
    ]
    metadata = [
        {
            "frame_id": "V001_00000_050",
            "video_id": "V001",
            "shot_id": 0,
            "timestamp": 1.0,
        }
    ]
    values = {
        "load_visual_embeddings": visual,
        "load_text_asr_embeddings": [],
        "load_text_summary_embeddings": [],
        "load_text_ocr_embeddings": [],
        "load_ocr_texts": [],
        "load_asr_transcripts": [],
        "load_video_summary": [],
        "load_metadata_and_objects": (metadata, []),
    }
    for name, value in values.items():
        monkeypatch.setattr(
            orchestrator_module,
            name,
            lambda *args, _value=value, **kwargs: _value,
        )

    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    milvus.snapshot_by_video_id.return_value = []
    es.snapshot_by_video_id.return_value = []
    tabular.snapshot_by_video_id.side_effect = [
        ([], []),
        (metadata, []),
    ]
    milvus.insert_batch.return_value = 1
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    assert not orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
    )

    # One delete for replacement and one for rollback cleanup.
    assert tabular.delete_by_video_id.call_count == 2
