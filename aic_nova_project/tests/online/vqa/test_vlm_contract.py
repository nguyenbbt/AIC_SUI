from __future__ import annotations

import pytest

from online.domain.errors import ContractMismatchError
from online.domain.vqa import ImageEvidence, VLMConfidence, VLMResponse, VLMResponseStatus, VQAAnswerType, VQAQuestion
from online.vqa.vlm_request import EVIDENCE_ONLY_INSTRUCTION, build_vlm_request, validate_vlm_response


QUESTION = VQAQuestion(question_id="q1", question="Có ai không?", answer_type=VQAAnswerType.YES_NO)
IMAGE = ImageEvidence(evidence_id="image-1", video_id="V001", frame_id="V001_00000_001", shot_id=0, timestamp_sec=1, image_reference="fixture://image/1")


def test_build_request_is_evidence_only_and_deterministic() -> None:
    request = build_vlm_request(QUESTION, (IMAGE,))
    assert request.request_id == "vlm-q1"
    assert request.evidence == (IMAGE,)
    lowered = EVIDENCE_ONLY_INSTRUCTION.lower()
    assert "external/world knowledge" in lowered
    assert "local path" in lowered
    assert "có ai không" not in lowered


def test_validate_answered_and_insufficient_responses() -> None:
    request = build_vlm_request(QUESTION, (IMAGE,))
    answered = VLMResponse(status=VLMResponseStatus.ANSWERED, answer="Có", answer_type=VQAAnswerType.YES_NO, confidence=VLMConfidence.HIGH, evidence_ids=(IMAGE.evidence_id,))
    insufficient = VLMResponse(status=VLMResponseStatus.INSUFFICIENT_EVIDENCE, answer_type=VQAAnswerType.YES_NO, confidence=VLMConfidence.LOW)
    assert validate_vlm_response(answered, request) is answered
    assert validate_vlm_response(insufficient.model_dump(), request) == insufficient


@pytest.mark.parametrize(
    "response",
    (
        object(),
        {"status": "answered"},
        {"status": "answered", "answer": "Có", "answer_type": "yes_no", "confidence": "high", "evidence_ids": ["unknown"]},
        {"status": "answered", "answer": "Có", "answer_type": "number", "confidence": "high", "evidence_ids": ["image-1"]},
    ),
)
def test_invalid_malformed_unknown_and_wrong_answer_type_are_rejected(response: object) -> None:
    with pytest.raises(ContractMismatchError):
        validate_vlm_response(response, build_vlm_request(QUESTION, (IMAGE,)))
