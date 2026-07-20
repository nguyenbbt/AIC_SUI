from __future__ import annotations

from fastapi.testclient import TestClient

from online.modes.vqa import VQAModeAdapter
from online.testing import build_advanced_modes_fixture
from online.vqa import EvidenceSelector, VQAOrchestrator
from retrieval_api.search_engine import create_app


class _CandidateHandoff:
    """Temporary structural handoff until Person B's Wave-3 adapter is merged."""

    def __init__(self, candidates) -> None:
        self._candidates = tuple(candidates)

    async def retrieve_candidates(self, question):
        return self._candidates


def test_vqa_internal_route_runs_selector_hydration_and_fake_vlm() -> None:
    fixture = build_advanced_modes_fixture()
    orchestrator = VQAOrchestrator(
        candidate_retriever=_CandidateHandoff(fixture.ranked_vqa_candidates),
        evidence_selector=EvidenceSelector(
            metadata_reader=fixture.metadata(),
            image_resolver=fixture.image_resolver(),
            evidence_hydrator=fixture.evidence_hydrator(),
        ),
        vlm=fixture.vlm(),
    )
    try:
        question = fixture.vqa_question
        response = TestClient(create_app(vqa_mode=VQAModeAdapter(orchestrator))).post(
            "/internal/unstable/vqa",
            json={
                "question_id": question.question_id,
                "question": question.question,
                "answer_type": question.answer_type.value,
            },
        )
    finally:
        orchestrator.close()

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["question_id"] == fixture.vqa_question.question_id
    assert result["response"]["status"] == "answered"
    assert set(result["response"]["evidence_ids"]).issubset(
        {item["evidence_id"] for item in result["evidence"]}
    )
