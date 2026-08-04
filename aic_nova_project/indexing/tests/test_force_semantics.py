from pathlib import Path
from unittest.mock import MagicMock

import src.indexing.orchestrator as orchestrator_module
from src.indexing.clients.milvus_client import VISUAL_COLLECTION
from src.indexing.orchestrator import IndexingOrchestrator


def _configure_core(monkeypatch):
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
    return visual, metadata


def test_force_controls_replacement_of_complete_snapshot(monkeypatch):
    visual, metadata = _configure_core(monkeypatch)
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    milvus.snapshot_by_video_id.side_effect = (
        lambda collection, video_id: (
            visual if collection == VISUAL_COLLECTION else []
        )
    )
    es.snapshot_by_video_id.return_value = []
    tabular.snapshot_by_video_id.return_value = (metadata, [])
    milvus.insert_batch.side_effect = (
        lambda collection, records, dimension: len(records)
    )
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    assert orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
        force=False,
    )
    milvus.delete_by_video_id.assert_not_called()
    milvus.insert_batch.assert_not_called()

    assert orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
        force=True,
    )
    assert milvus.delete_by_video_id.called
    assert milvus.insert_batch.called


def test_run_forwards_force_to_each_video(tmp_path, monkeypatch):
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    orchestrator = IndexingOrchestrator(milvus, es, tabular)
    orchestrator.process_video = MagicMock(return_value=True)
    monkeypatch.setattr(
        orchestrator_module,
        "discover_video_ids",
        lambda data_dir: ["V001"],
    )
    monkeypatch.setattr(
        orchestrator_module,
        "detect_embedding_dim",
        lambda directory: 2,
    )

    orchestrator.run(tmp_path, force=True)

    assert orchestrator.process_video.call_args.kwargs["force"] is True
