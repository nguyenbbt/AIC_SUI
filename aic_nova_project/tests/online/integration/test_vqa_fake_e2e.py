from __future__ import annotations

import asyncio

import pytest

from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.domain.vqa import ASREvidence, VLMResponseStatus
from online.testing import (
    AdvancedFakeBehavior,
    FakeImageResolver,
    FakeVLMMode,
    build_advanced_modes_fixture,
)
from online.vqa import EvidenceSelector, VQAOrchestrator


class _FixtureCandidateRetriever:
    def __init__(self, candidates) -> None:
        self._candidates = tuple(candidates)

    async def retrieve_candidates(self, question):
        return self._candidates


def _orchestrator(fixture, *, image_resolver=None, evidence_hydrator=None, vlm=None):
    selector = EvidenceSelector(
        metadata_reader=fixture.metadata(),
        image_resolver=image_resolver or fixture.image_resolver(),
        evidence_hydrator=evidence_hydrator or fixture.evidence_hydrator(),
    )
    return VQAOrchestrator(
        candidate_retriever=_FixtureCandidateRetriever(
            fixture.ranked_vqa_candidates
        ),
        evidence_selector=selector,
        vlm=vlm or fixture.vlm(),
    )


def _diagnostics_without_latency(result) -> dict[str, object]:
    values = result.diagnostics.model_dump()
    values.pop("vlm_latency_ms")
    return values


def test_vqa_shared_fixture_answered_path_is_grounded_and_deterministic() -> None:
    fixture = build_advanced_modes_fixture()

    async def scenario():
        service = _orchestrator(fixture)
        try:
            first = await service.answer(fixture.vqa_question)
            second = await service.answer(fixture.vqa_question)
            return first, second
        finally:
            service.close()

    first, second = asyncio.run(scenario())
    actual_ids = tuple(item.evidence_id for item in first.evidence)

    assert actual_ids == fixture.expected_vqa_selected_evidence_ids
    assert first.response.status is VLMResponseStatus.ANSWERED
    assert first.response.evidence_ids == fixture.expected_vqa_answer_evidence_ids
    assert set(first.response.evidence_ids).issubset(actual_ids)
    assert first.question_id == second.question_id
    assert first.response == second.response
    assert first.evidence == second.evidence
    assert _diagnostics_without_latency(first) == _diagnostics_without_latency(second)


def test_vqa_shared_fixture_insufficient_and_missing_image_paths_do_not_fabricate() -> None:
    fixture = build_advanced_modes_fixture()

    async def insufficient_scenario():
        service = _orchestrator(
            fixture,
            vlm=fixture.vlm(FakeVLMMode.INSUFFICIENT),
        )
        try:
            return await service.answer(fixture.vqa_question)
        finally:
            service.close()

    insufficient = asyncio.run(insufficient_scenario())
    assert insufficient.response.status is VLMResponseStatus.INSUFFICIENT_EVIDENCE
    assert insufficient.response.answer is None
    assert insufficient.response.evidence_ids == ()

    missing_vlm = fixture.vlm()

    async def missing_images_scenario():
        service = _orchestrator(
            fixture,
            image_resolver=FakeImageResolver({}),
            vlm=missing_vlm,
        )
        try:
            return await service.answer(fixture.vqa_question)
        finally:
            service.close()

    missing = asyncio.run(missing_images_scenario())
    assert missing.response.status is VLMResponseStatus.INSUFFICIENT_EVIDENCE
    assert missing.response.answer is None
    assert missing.diagnostics.selected_image_count == 0
    assert "NO_IMAGE_EVIDENCE" in missing.diagnostics.warnings
    assert missing_vlm.calls == ()


def test_vqa_shared_fixture_degraded_text_and_malformed_paths_are_bounded() -> None:
    fixture = build_advanced_modes_fixture()
    primary_image_id = fixture.expected_vqa_selected_evidence_ids[0]
    degraded_vlm = fixture.vlm(grounded_evidence_ids=(primary_image_id,))
    degraded_hydrator = fixture.evidence_hydrator(
        behaviors={
            "asr": AdvancedFakeBehavior(
                ResourceUnavailableError("simulated ASR outage")
            )
        }
    )

    async def degraded_scenario():
        service = _orchestrator(
            fixture,
            evidence_hydrator=degraded_hydrator,
            vlm=degraded_vlm,
        )
        try:
            return await service.answer(fixture.vqa_question)
        finally:
            service.close()

    degraded = asyncio.run(degraded_scenario())
    assert degraded.response.status is VLMResponseStatus.ANSWERED
    assert "ASR_RESOURCE_UNAVAILABLE" in degraded.diagnostics.warnings
    assert all(not isinstance(item, ASREvidence) for item in degraded.evidence)

    malformed_vlm = fixture.vlm(FakeVLMMode.MALFORMED)

    async def malformed_scenario():
        service = _orchestrator(fixture, vlm=malformed_vlm)
        try:
            with pytest.raises(ContractMismatchError):
                await service.answer(fixture.vqa_question)
        finally:
            service.close()

    asyncio.run(malformed_scenario())
    assert len(malformed_vlm.calls) == 2

