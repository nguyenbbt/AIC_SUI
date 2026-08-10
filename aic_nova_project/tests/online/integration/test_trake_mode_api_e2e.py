from __future__ import annotations

from fastapi.testclient import TestClient

from online.modes.trake import TRAKEModeAdapter
from online.testing import build_advanced_modes_fixture
from online.trake import TRAKEService
from retrieval_api.search_engine import create_app


def test_trake_internal_route_runs_service_and_dante_with_ordered_events() -> None:
    fixture = build_advanced_modes_fixture()
    service = TRAKEService(corpus=fixture.visual_corpus(), encoder=fixture.text_encoder())
    try:
        payload = {
            "query_id": fixture.trake_query.query_id,
            "event_ids": [event.event_id for event in fixture.trake_query.events],
            "event_texts": [event.text for event in fixture.trake_query.events],
            "top_k_videos": fixture.trake_query.top_k_videos,
            "policy": fixture.trake_query.policy.model_dump(),
        }
        response = TestClient(create_app(trake_mode=TRAKEModeAdapter(service))).post(
            "/internal/unstable/trake", json=payload
        )
    finally:
        service.close()

    assert response.status_code == 200
    body = response.json()
    assert body["query_id"] == fixture.trake_query.query_id
    assert body["results"][0]["video_id"] == fixture.expected_dante_video_id
    assert [item["event_id"] for item in body["results"][0]["sequence"]] == payload["event_ids"]
