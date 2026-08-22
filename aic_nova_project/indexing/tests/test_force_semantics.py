from pathlib import Path
from unittest.mock import MagicMock

import src.indexing.orchestrator as orchestrator_module
import pytest
from src.indexing.clients.milvus_client import VISUAL_COLLECTION
from src.indexing.orchestrator import IndexingOrchestrator
from tests.contract_fixtures import canonical_video_record


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
    tabular.snapshot_video_by_id.return_value = canonical_video_record(
        "V001"
    )
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


def test_bulk_rebuild_requires_reset_all(tmp_path):
    orchestrator = IndexingOrchestrator(
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    with pytest.raises(ValueError, match="reset_all"):
        orchestrator.run(tmp_path, bulk_rebuild=True)


def test_bulk_rebuild_skips_per_video_snapshot_and_flush(monkeypatch):
    _configure_core(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "load_video_metadata",
        lambda *args, **kwargs: canonical_video_record("V001"),
    )
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    milvus.insert_batch.return_value = 1
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    assert orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
        bulk_rebuild=True,
    )

    milvus.snapshot_by_video_id.assert_not_called()
    milvus.delete_by_video_id.assert_not_called()
    milvus.insert_batch.assert_called_once_with(
        VISUAL_COLLECTION,
        orchestrator_module.load_visual_embeddings(Path("/fake"), "V001"),
        2,
        flush=False,
    )


def test_bulk_rebuild_finalizes_visibility_once(tmp_path, monkeypatch):
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    orchestrator = IndexingOrchestrator(milvus, es, tabular)
    orchestrator.process_video = MagicMock(return_value=True)
    monkeypatch.setattr(
        orchestrator_module,
        "discover_video_ids",
        lambda data_dir: ["V001", "V002"],
    )
    monkeypatch.setattr(
        orchestrator_module,
        "detect_embedding_dim",
        lambda directory: 2,
    )

    orchestrator.run(
        tmp_path,
        reset_all=True,
        bulk_rebuild=True,
    )

    assert all(
        call.kwargs["bulk_rebuild"] is True
        for call in orchestrator.process_video.call_args_list
    )
    milvus.flush_collections.assert_called_once()
    es.refresh_indices.assert_called_once()


def test_targeted_repair_only_processes_requested_videos_and_finalizes(
    tmp_path,
    monkeypatch,
):
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    orchestrator = IndexingOrchestrator(milvus, es, tabular)
    orchestrator.process_video = MagicMock(return_value=True)
    monkeypatch.setattr(
        orchestrator_module,
        "discover_video_ids",
        lambda data_dir: ["L26_V307", "L26_V308", "L26_V309"],
    )
    monkeypatch.setattr(
        orchestrator_module,
        "detect_embedding_dim",
        lambda directory: 2,
    )

    orchestrator.run(
        tmp_path,
        video_ids=["L26_V308", "L26_V309"],
        finalize=True,
    )

    assert [
        call.args[0] for call in orchestrator.process_video.call_args_list
    ] == ["L26_V308", "L26_V309"]
    milvus.flush_collections.assert_called_once()
    es.refresh_indices.assert_called_once()


def test_targeted_repair_rejects_unknown_video_before_processing(
    tmp_path,
    monkeypatch,
):
    orchestrator = IndexingOrchestrator(
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    orchestrator.process_video = MagicMock(return_value=True)
    monkeypatch.setattr(
        orchestrator_module,
        "discover_video_ids",
        lambda data_dir: ["L26_V308"],
    )
    monkeypatch.setattr(
        orchestrator_module,
        "detect_embedding_dim",
        lambda directory: 2,
    )

    with pytest.raises(ValueError, match="L26_V999"):
        orchestrator.run(tmp_path, video_ids=["L26_V999"])

    orchestrator.process_video.assert_not_called()


def test_unpublished_repair_skips_snapshot_but_deletes_partial_records(
    monkeypatch,
):
    _configure_core(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "load_video_metadata",
        lambda *args, **kwargs: canonical_video_record("V001"),
    )
    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    milvus.insert_batch.return_value = 1
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    assert orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
        unpublished_repair=True,
    )

    milvus.snapshot_by_video_id.assert_not_called()
    milvus.delete_by_video_id.assert_called()


def test_unpublished_repair_requires_selected_videos_and_finalize(tmp_path):
    orchestrator = IndexingOrchestrator(
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    with pytest.raises(ValueError, match="selected video IDs"):
        orchestrator.run(tmp_path, unpublished_repair=True, finalize=True)
    with pytest.raises(ValueError, match="finalize"):
        orchestrator.run(
            tmp_path,
            video_ids=["V001"],
            unpublished_repair=True,
        )
