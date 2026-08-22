from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.indexing.orchestrator as orchestrator_module
from src.indexing.clients.es_client import ESClient, OCR_INDEX
from src.indexing.orchestrator import IndexingOrchestrator


@pytest.mark.parametrize(
    ("bulk_result", "message"),
    [
        ((1, [{"index": {"error": "rejected"}}]), "1 error"),
        ((1, []), "1/2"),
    ],
)
def test_es_bulk_requires_every_document_to_succeed(
    bulk_result,
    message,
):
    client = ESClient()
    client.client = MagicMock()
    documents = [
        {"frame_id": "f1", "video_id": "V001"},
        {"frame_id": "f2", "video_id": "V001"},
    ]

    with patch(
        "src.indexing.clients.es_client.bulk",
        return_value=bulk_result,
    ):
        with pytest.raises(RuntimeError, match=message):
            client.bulk_index(OCR_INDEX, documents, id_field="frame_id")


def test_es_bulk_waits_until_documents_are_search_visible():
    client = ESClient()
    client.client = MagicMock()
    documents = [
        {"frame_id": "f1", "video_id": "V001"},
    ]

    with patch(
        "src.indexing.clients.es_client.bulk",
        return_value=(1, []),
    ) as bulk_mock:
        client.bulk_index(OCR_INDEX, documents, id_field="frame_id")

    assert bulk_mock.call_args.kwargs["refresh"] == "wait_for"


def test_es_bulk_can_defer_refresh_for_fresh_rebuild():
    client = ESClient()
    client.client = MagicMock()
    documents = [{"frame_id": "f1", "video_id": "V001"}]

    with patch(
        "src.indexing.clients.es_client.bulk",
        return_value=(1, []),
    ) as bulk_mock:
        client.bulk_index(
            OCR_INDEX,
            documents,
            id_field="frame_id",
            refresh=False,
        )

    assert bulk_mock.call_args.kwargs["refresh"] is False


def test_orchestrator_rejects_partial_bulk_count(monkeypatch):
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
        "load_ocr_texts": [
            {
                "frame_id": "V001_00000_050",
                "video_id": "V001",
                "shot_id": "0",
                "ocr_text_concat": "text",
            }
        ],
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

    milvus = MagicMock()
    es = MagicMock()
    tabular = MagicMock()
    es.bulk_index.return_value = 0
    orchestrator = IndexingOrchestrator(milvus, es, tabular)

    assert not orchestrator.process_video(
        "V001",
        Path("/fake"),
        visual_dim=2,
        text_dim=2,
    )
