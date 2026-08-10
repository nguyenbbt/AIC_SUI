from pathlib import Path

import src.indexing.orchestrator as orchestrator_module
from src.indexing.clients.es_client import ASR_INDEX, OCR_INDEX, SUMMARY_INDEX
from src.indexing.clients.milvus_client import (
    ASR_COLLECTION,
    OCR_COLLECTION,
    SUMMARY_COLLECTION,
    VISUAL_COLLECTION,
)
from src.indexing.clients.tabular_client import TabularClient
from src.indexing.orchestrator import IndexingOrchestrator


class InMemoryMilvus:
    def __init__(self):
        self.records = {
            VISUAL_COLLECTION: [],
            ASR_COLLECTION: [],
            SUMMARY_COLLECTION: [],
            OCR_COLLECTION: [],
        }

    def snapshot_by_video_id(self, collection, video_id):
        return [
            dict(record)
            for record in self.records[collection]
            if record["video_id"] == video_id
        ]

    def delete_by_video_id(self, collection, video_id):
        self.records[collection] = [
            record
            for record in self.records[collection]
            if record["video_id"] != video_id
        ]

    def insert_batch(self, collection, records, dimension):
        self.records[collection].extend(dict(record) for record in records)
        return len(records)


class InMemoryElasticsearch:
    def __init__(self):
        self.documents = {
            OCR_INDEX: [],
            ASR_INDEX: [],
            SUMMARY_INDEX: [],
        }

    def snapshot_by_video_id(self, index_name, video_id):
        return [
            dict(document)
            for document in self.documents[index_name]
            if document.get("_source", {}).get("video_id") == video_id
        ]

    def delete_by_video_id(self, index_name, video_id):
        self.documents[index_name] = [
            document
            for document in self.documents[index_name]
            if document.get("_source", {}).get("video_id") != video_id
        ]

    def restore_snapshot(self, index_name, documents):
        self.documents[index_name].extend(documents)


def test_process_video_persists_video_before_frame_metadata(
    tmp_path,
    monkeypatch,
):
    video_id = "V001"
    video = {
        "video_id": video_id,
        "source_video_rel_path": "raw_videos/V001.mp4",
        "fps": 25.0,
        "duration_sec": 10.0,
        "frame_count": 250,
        "width": 1280,
        "height": 720,
    }
    visual = [
        {
            "frame_id": "V001_00000_050",
            "video_id": video_id,
            "shot_id": 0,
            "embedding": [1.0, 0.0],
        }
    ]
    metadata = [
        {
            "frame_id": "V001_00000_050",
            "video_id": video_id,
            "shot_id": 0,
            "source_frame_idx": 125,
            "timestamp": 5.0,
            "image_rel_path": "keyframes/V001/shot_00000_pos_050.webp",
        }
    ]
    loader_values = {
        "load_video_metadata": video,
        "load_visual_embeddings": visual,
        "load_text_asr_embeddings": [],
        "load_text_summary_embeddings": [],
        "load_text_ocr_embeddings": [],
        "load_ocr_texts": [],
        "load_asr_transcripts": [],
        "load_video_summary": [],
        "load_metadata_and_objects": (metadata, []),
    }
    for name, value in loader_values.items():
        monkeypatch.setattr(
            orchestrator_module,
            name,
            lambda *args, _value=value, **kwargs: _value,
        )

    tabular = TabularClient(str(tmp_path / "metadata.db"))
    tabular.connect()
    tabular.create_tables()
    orchestrator = IndexingOrchestrator(
        InMemoryMilvus(),
        InMemoryElasticsearch(),
        tabular,
    )

    assert orchestrator.process_video(
        video_id,
        Path("/unused"),
        visual_dim=2,
        text_dim=None,
    )
    assert tabular.snapshot_video_by_id(video_id) == video
    assert tabular.snapshot_by_video_id(video_id)[0] == metadata
    assert tabular.conn.execute("PRAGMA foreign_key_check").fetchall() == []

    tabular.disconnect()
