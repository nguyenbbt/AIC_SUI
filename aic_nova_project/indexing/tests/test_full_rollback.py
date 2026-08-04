from pathlib import Path
from unittest.mock import MagicMock

import src.indexing.orchestrator as orchestrator_module
from src.indexing.clients.es_client import OCR_INDEX
from src.indexing.clients.milvus_client import VISUAL_COLLECTION
from src.indexing.orchestrator import IndexingOrchestrator


def _set_loaders(monkeypatch, **overrides):
    values = {
        "load_visual_embeddings": [
            {
                "frame_id": "V001_00000_050",
                "video_id": "V001",
                "shot_id": 0,
                "embedding": [1.0, 0.0],
            }
        ],
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
        **overrides,
    }
    for name, value in values.items():
        monkeypatch.setattr(
            orchestrator_module,
            name,
            lambda *args, _value=value, **kwargs: _value,
        )


def _clients_with_snapshots():
    old_visual = {
        "frame_id": "V001_00000_050",
        "video_id": "V001",
        "shot_id": 0,
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
    milvus.insert_batch.side_effect = (
        lambda collection, records, dimension: len(records)
    )
    return milvus, es, tabular, old_visual, old_ocr, old_metadata


def test_es_failure_in_later_batch_cleans_partial_and_restores_all(
    monkeypatch,
):
    ocr_records = [
        {
            "frame_id": f"V001_0000{i}_050",
            "video_id": "V001",
            "shot_id": str(i),
            "ocr_text_concat": f"text {i}",
        }
        for i in range(3)
    ]
    _set_loaders(monkeypatch, load_ocr_texts=ocr_records)
    (
        milvus,
        es,
        tabular,
        old_visual,
        old_ocr,
        old_metadata,
    ) = _clients_with_snapshots()
    es.bulk_index.side_effect = [2, RuntimeError("second batch failed")]
    orchestrator = IndexingOrchestrator(
        milvus,
        es,
        tabular,
        batch_size=2,
    )

    assert not orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
    )

    assert es.delete_by_video_id.call_count == 6
    assert tabular.delete_by_video_id.call_count == 2
    es.restore_snapshot.assert_called_once_with(OCR_INDEX, [old_ocr])
    tabular.restore_snapshot.assert_called_once_with(old_metadata, [])
    assert any(
        call.args[1] == [old_visual]
        for call in milvus.insert_batch.call_args_list
    )


def test_sqlite_failure_in_later_batch_restores_old_rows(monkeypatch):
    new_metadata = [
        {
            "frame_id": f"V001_0000{i}_050",
            "video_id": "V001",
            "shot_id": i,
            "timestamp": float(i),
        }
        for i in range(2)
    ]
    _set_loaders(
        monkeypatch,
        load_metadata_and_objects=(new_metadata, []),
    )
    (
        milvus,
        es,
        tabular,
        old_visual,
        old_ocr,
        old_metadata,
    ) = _clients_with_snapshots()
    tabular.insert_metadata_batch.side_effect = [
        None,
        RuntimeError("second SQLite batch failed"),
    ]
    orchestrator = IndexingOrchestrator(
        milvus,
        es,
        tabular,
        batch_size=1,
    )

    assert not orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
    )

    assert tabular.delete_by_video_id.call_count == 2
    tabular.restore_snapshot.assert_called_once_with(old_metadata, [])
    es.restore_snapshot.assert_called_once_with(OCR_INDEX, [old_ocr])
    assert any(
        call.args[1] == [old_visual]
        for call in milvus.insert_batch.call_args_list
    )
