from __future__ import annotations

import pytest

from online.domain.errors import ContractMismatchError, InvalidQueryError
from online.domain.vqa import VLMConfidence, VLMResponse, VLMResponseStatus
from online.testing import build_advanced_modes_fixture
from retrieval_api.submission import (
    serialize_kis_submissions,
    serialize_trake_submissions,
    serialize_vqa_submission,
)


def test_aic2026_logical_submission_rows_use_original_video_frame_indices() -> None:
    fixture = build_advanced_modes_fixture()
    kis = serialize_kis_submissions(fixture.ranked_vqa_candidates)
    assert kis[0].model_dump(mode="json") == {
        "video_id": fixture.ranked_vqa_candidates[0].video_id,
        "frame_id": fixture.ranked_vqa_candidates[0].source_frame_idx,
    }

    image = next(iter(fixture.images_by_frame_id.values()))
    response = VLMResponse(
        status=VLMResponseStatus.ANSWERED,
        answer="5",
        answer_type=fixture.vqa_question.answer_type,
        confidence=VLMConfidence.HIGH,
        evidence_ids=(image.evidence_id,),
    )
    vqa = serialize_vqa_submission(image=image, response=response)
    assert vqa.model_dump(mode="json") == {
        "video_id": image.video_id,
        "frame_id": image.source_frame_idx,
        "answer": "5",
    }


def test_trake_submission_preserves_event_order() -> None:
    import asyncio

    from online.trake import TRAKEService

    fixture = build_advanced_modes_fixture()
    service = TRAKEService(corpus=fixture.visual_corpus(), encoder=fixture.text_encoder())
    execution = asyncio.run(service.execute(fixture.trake_query))
    service.close()

    rows = serialize_trake_submissions(execution.results)

    assert rows[0].video_id == execution.results[0].video_id
    assert rows[0].frame_ids == tuple(
        match.source_frame_idx for match in execution.results[0].sequence
    )


def test_submission_contract_enforces_cap_and_answered_vqa() -> None:
    fixture = build_advanced_modes_fixture()
    with pytest.raises(InvalidQueryError):
        serialize_kis_submissions(fixture.ranked_vqa_candidates, limit=101)
    image = next(iter(fixture.images_by_frame_id.values()))
    response = VLMResponse(
        status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
        answer_type=fixture.vqa_question.answer_type,
        confidence=VLMConfidence.LOW,
    )
    with pytest.raises(ContractMismatchError):
        serialize_vqa_submission(image=image, response=response)
