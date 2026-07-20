from __future__ import annotations

from fastapi.testclient import TestClient

from online.domain.trake import TRAKEDiagnostics
from online.domain.vqa import VLMConfidence, VLMResponse, VLMResponseStatus, VQAResult
from online.modes.trake import TRAKEModeAdapter
from online.modes.vqa import VQAModeAdapter
from online.trake.service import TRAKEExecution
from retrieval_api.search_engine import create_app


class _TRAKE:
    async def execute(self, query):
        return TRAKEExecution(
            query_id=query.query_id,
            results=(),
            diagnostics=TRAKEDiagnostics(
                policy_version=query.policy.policy_version,
                lambda_penalty=query.policy.lambda_penalty,
                event_count=len(query.events),
                video_count=0,
                frame_count=0,
            ),
        )


class _VQA:
    async def answer(self, question, budget):
        return VQAResult(
            question_id=question.question_id,
            response=VLMResponse(
                status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
                answer_type=question.answer_type,
                confidence=VLMConfidence.LOW,
            ),
        )


def test_advanced_routes_success_and_preserve_ids() -> None:
    client = TestClient(
        create_app(trake_mode=TRAKEModeAdapter(_TRAKE()), vqa_mode=VQAModeAdapter(_VQA()))
    )
    trake = client.post(
        "/internal/unstable/trake",
        json={"query_id": "t1", "event_texts": ["first", "second"]},
    )
    assert trake.status_code == 200
    assert trake.json()["query_id"] == "t1"
    vqa = client.post(
        "/internal/unstable/vqa",
        json={"question_id": "q1", "question": "What?", "answer_type": "short_text"},
    )
    assert vqa.status_code == 200
    assert vqa.json()["question_id"] == "q1"
    assert vqa.json()["result"]["response"]["status"] == "insufficient_evidence"


def test_advanced_routes_invalid_and_disabled() -> None:
    client = TestClient(create_app())
    assert client.post("/internal/unstable/trake", json={}).status_code == 422
    response = client.post(
        "/internal/unstable/vqa",
        json={"question_id": "q1", "question": "What?", "answer_type": "short_text"},
    )
    assert response.status_code == 503
    assert "vqa_mode" not in response.text
