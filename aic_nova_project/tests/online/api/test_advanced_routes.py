from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from online.domain.errors import BranchTimeoutError, ResourceUnavailableError
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


class _FailingTRAKE:
    def __init__(self, error) -> None:
        self.error = error

    async def execute(self, query):
        raise self.error


class _ConcurrentVQA(_VQA):
    async def answer(self, question, budget):
        time.sleep(0.005 if question.question_id.endswith("1") else 0.001)
        return await super().answer(question, budget)


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


@pytest.mark.parametrize(
    ("error", "expected_status"),
    (
        (BranchTimeoutError("secret timeout details"), 504),
        (ResourceUnavailableError("secret backend details"), 503),
    ),
)
def test_trake_route_maps_typed_failures_without_leaking_details(error, expected_status) -> None:
    response = TestClient(create_app(trake_mode=_FailingTRAKE(error))).post(
        "/internal/unstable/trake",
        json={"query_id": "t1", "event_texts": ["first", "second"]},
    )
    assert response.status_code == expected_status
    assert "secret" not in response.text


def test_unexpected_advanced_failure_is_sanitized() -> None:
    response = TestClient(
        create_app(trake_mode=_FailingTRAKE(RuntimeError("token=secret"))),
        raise_server_exceptions=False,
    ).post(
        "/internal/unstable/trake",
        json={"query_id": "t1", "event_texts": ["first", "second"]},
    )
    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "The service could not complete the request",
            "details": {},
        }
    }


def test_concurrent_vqa_routes_keep_question_ids_isolated() -> None:
    client = TestClient(create_app(vqa_mode=_ConcurrentVQA()))

    def request(question_id: str):
        return client.post(
            "/internal/unstable/vqa",
            json={
                "question_id": question_id,
                "question": "What?",
                "answer_type": "short_text",
            },
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = tuple(executor.map(request, ("q1", "q2")))
    assert tuple(response.status_code for response in responses) == (200, 200)
    assert tuple(response.json()["question_id"] for response in responses) == ("q1", "q2")
