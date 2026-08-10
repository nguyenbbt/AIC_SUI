from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from src.indexing.clients.es_client import (
    ESClient,
    OCR_INDEX,
    OCR_MAPPING,
)
from src.indexing.clients.milvus_client import (
    ASR_COLLECTION,
    HNSW_INDEX_PARAMS,
    OCR_COLLECTION,
    SUMMARY_COLLECTION,
    VISUAL_COLLECTION,
    MilvusVectorClient,
    _build_visual_schema,
)
from src.indexing.clients.tabular_client import TabularClient
from src.indexing.orchestrator import IndexingOrchestrator


def test_existing_milvus_collection_dimension_is_audited():
    collection = MagicMock()
    collection.schema = _build_visual_schema(3)
    collection.indexes = [
        SimpleNamespace(
            field_name="embedding",
            params=HNSW_INDEX_PARAMS,
        )
    ]
    client = MilvusVectorClient()

    with patch(
        "src.indexing.clients.milvus_client.utility.has_collection",
        return_value=True,
    ), patch(
        "src.indexing.clients.milvus_client.Collection",
        return_value=collection,
    ):
        with pytest.raises(ValueError, match="schema"):
            client.create_collection_if_not_exists(
                VISUAL_COLLECTION,
                dim=2,
            )


def test_existing_es_mapping_is_audited():
    client = ESClient()
    client.client = MagicMock()
    client.client.indices.exists.return_value = True
    client.client.indices.get_mapping.return_value = {
        OCR_INDEX: {
            "mappings": {
                "properties": {
                    "frame_id": {"type": "text"},
                }
            }
        }
    }

    with pytest.raises(ValueError, match="mapping"):
        client._create_index_if_not_exists(OCR_INDEX, OCR_MAPPING)


def test_existing_sqlite_schema_is_audited():
    client = TabularClient(":memory:")
    client.connect()
    client.conn.execute(
        "CREATE TABLE metadata ("
        "frame_id TEXT PRIMARY KEY, "
        "video_id TEXT NOT NULL, "
        "shot_id INTEGER NOT NULL, "
        "timestamp TEXT NOT NULL)"
    )
    client.conn.commit()

    with pytest.raises(ValueError, match="SQLite schema"):
        client.create_tables()

    client.disconnect()


def test_sqlite_self_indexed_v2_schema_and_video_join():
    client = TabularClient(":memory:")
    client.connect()
    client.create_tables()

    client.insert_video_batch(
        [
            {
                "video_id": "V001",
                "source_video_rel_path": "raw_videos/V001.mp4",
                "fps": 25.0,
                "duration_sec": 10.0,
                "frame_count": 250,
                "width": 1280,
                "height": 720,
            }
        ]
    )
    client.insert_metadata_batch(
        [
            {
                "frame_id": "V001_00000_050",
                "video_id": "V001",
                "shot_id": 0,
                "source_frame_idx": 125,
                "timestamp": 5.0,
                "image_rel_path": "keyframes/V001/shot_00000_pos_050.webp",
            }
        ]
    )

    tables = {
        row[0]
        for row in client.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    metadata_columns = {
        row[1] for row in client.conn.execute("PRAGMA table_info(metadata)")
    }
    metadata_indexes = {
        row[1] for row in client.conn.execute("PRAGMA index_list(metadata)")
    }
    assert "videos" in tables
    assert {"source_frame_idx", "image_rel_path"} <= metadata_columns
    assert {
        "idx_metadata_video_id",
        "idx_metadata_video_timeline",
        "idx_metadata_video_source_frame",
    } <= metadata_indexes
    assert client.conn.execute("PRAGMA foreign_key_check").fetchall() == []

    client.disconnect()


def test_run_provisions_or_audits_all_vector_collections(
    tmp_path,
):
    for relative_path in (
        "embeddings/visual",
        "embeddings/text_asr",
        "embeddings/text_ocr",
    ):
        (tmp_path / relative_path).mkdir(parents=True)

    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    with patch(
        "src.indexing.orchestrator.detect_embedding_dim",
        return_value=2,
    ), patch(
        "src.indexing.orchestrator.discover_video_ids",
        return_value=[],
    ):
        orchestrator.run(tmp_path)

    assert milvus.create_collection_if_not_exists.call_args_list == [
        call(VISUAL_COLLECTION, 2),
        call(ASR_COLLECTION, 2),
        call(SUMMARY_COLLECTION, 2),
        call(OCR_COLLECTION, 2),
    ]
