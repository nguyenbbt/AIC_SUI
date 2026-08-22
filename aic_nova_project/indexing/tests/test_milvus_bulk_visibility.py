from unittest.mock import MagicMock

from src.indexing.clients.milvus_client import (
    MilvusVectorClient,
    VISUAL_COLLECTION,
)


def test_milvus_insert_can_defer_flush_for_fresh_rebuild():
    client = MilvusVectorClient()
    collection = MagicMock()
    collection.insert.return_value.insert_count = 1
    client.create_collection_if_not_exists = MagicMock(
        return_value=collection
    )

    inserted = client.insert_batch(
        VISUAL_COLLECTION,
        [
            {
                "frame_id": "V001_00000_050",
                "video_id": "V001",
                "shot_id": 0,
                "embedding": [1.0, 0.0],
            }
        ],
        2,
        flush=False,
    )

    assert inserted == 1
    collection.flush.assert_not_called()
