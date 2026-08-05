from pathlib import Path
from unittest.mock import MagicMock

import src.indexing.orchestrator as orchestrator_module
from src.indexing.clients.es_client import OCR_INDEX
from src.indexing.clients.milvus_client import VISUAL_COLLECTION
from src.indexing.orchestrator import IndexingOrchestrator


def _patch_loaders(monkeypatch, *, visual_records):
    values = {
        "load_visual_embeddings": visual_records,
        "load_text_asr_embeddings": [],
        "load_text_summary_embeddings": [],
        "load_text_ocr_embeddings": [],
        "load_ocr_texts": [],
        "load_asr_transcripts": [],
        "load_video_summary": [],
        "load_metadata_and_objects": (
            [
                {
                    "frame_id": "V001_00000_050",
                    "video_id": "V001",
                    "shot_id": 0,
                    "timestamp": 1.0,
                }
            ],
            [],
        ),
    }
    for name, value in values.items():
        monkeypatch.setattr(
            orchestrator_module,
            name,
            lambda *args, _value=value, **kwargs: _value,
        )


def test_snapshot_failure_never_deletes_existing_data(monkeypatch):
    _patch_loaders(
        monkeypatch,
        visual_records=[
            {
                "frame_id": "V001_00000_050",
                "video_id": "V001",
                "shot_id": 0,
                "embedding": [1.0, 0.0],
            }
        ],
    )
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    milvus.snapshot_by_video_id.side_effect = RuntimeError(
        "snapshot unavailable"
    )
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    assert not orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
    )

    milvus.delete_by_video_id.assert_not_called()
    es.delete_by_video_id.assert_not_called()
    tabular.delete_by_video_id.assert_not_called()


def test_failed_replace_restores_last_known_good_snapshot(monkeypatch):
    new_visual = {
        "frame_id": "V001_00000_050",
        "video_id": "V001",
        "shot_id": 0,
        "embedding": [1.0, 0.0],
    }
    old_visual = {
        **new_visual,
        "embedding": [0.0, 1.0],
    }
    old_ocr = {
        "_id": "V001_00000_050",
        "_source": {
            "frame_id": "V001_00000_050",
            "video_id": "V001",
            "shot_id": "0",
            "ocr_text_concat": "old",
        },
    }
    old_metadata = [
        {
            "frame_id": "V001_00000_050",
            "video_id": "V001",
            "shot_id": 0,
            "timestamp": 1.0,
        }
    ]
    _patch_loaders(monkeypatch, visual_records=[new_visual])

    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    milvus.snapshot_by_video_id.side_effect = (
        lambda collection, video_id: (
            [old_visual]
            if collection == VISUAL_COLLECTION
            else []
        )
    )
    es.snapshot_by_video_id.side_effect = (
        lambda index, video_id: [old_ocr] if index == OCR_INDEX else []
    )
    tabular.snapshot_by_video_id.return_value = (old_metadata, [])

    def insert_batch(collection, records, dimension):
        if records == [new_visual]:
            raise RuntimeError("new insert failed")
        return len(records)

    milvus.insert_batch.side_effect = insert_batch
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    assert not orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
    )

    assert any(
        call.args[1] == [old_visual]
        for call in milvus.insert_batch.call_args_list
    )
    es.restore_snapshot.assert_called_once_with(OCR_INDEX, [old_ocr])
    tabular.restore_snapshot.assert_called_once_with(old_metadata, [])
