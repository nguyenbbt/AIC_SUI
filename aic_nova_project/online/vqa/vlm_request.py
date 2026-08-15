"""Evidence-only VLM request construction and defensive response validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from online.domain.errors import ContractMismatchError
from online.domain.vqa import VQAEvidence, VLMRequest, VLMResponse, VQAQuestion


EVIDENCE_ONLY_INSTRUCTION = """Chỉ trả lời từ evidence được cung cấp.
Không dùng external/world knowledge để điền phần thiếu.
Evidence IDs phải là subset của request.
Trả lời ngắn gọn cùng ngôn ngữ với câu hỏi.
Không trả chain-of-thought, secret hoặc local path.
Return compact JSON without extra whitespace.
answer_type MUST exactly match the question answer_type.
If status is answered, answer MUST be a non-empty answer and evidence_ids MUST
contain at least one supplied evidence ID.
If status is insufficient_evidence, answer MUST be null and evidence_ids MUST be [].
Never write the literal string insufficient_evidence into answer."""


def build_vlm_request(
    question: VQAQuestion,
    evidence: Sequence[VQAEvidence],
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 512,
) -> VLMRequest:
    if not isinstance(question, VQAQuestion):
        raise ContractMismatchError("question must be a validated VQAQuestion")
    return VLMRequest(
        request_id=f"vlm-{question.question_id}",
        question=question,
        evidence=tuple(evidence),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def validate_vlm_response(response: object, request: VLMRequest) -> VLMResponse:
    if not isinstance(request, VLMRequest):
        raise ContractMismatchError("request must be a validated VLMRequest")
    try:
        parsed = (
            response
            if isinstance(response, VLMResponse)
            else VLMResponse.model_validate(response)
            if isinstance(response, Mapping)
            else None
        )
    except ValidationError as exc:
        raise ContractMismatchError("VLM returned a protocol-invalid response") from exc
    if parsed is None:
        raise ContractMismatchError("VLM returned a non-VLMResponse value")
    allowed_ids = {item.evidence_id for item in request.evidence}
    if not set(parsed.evidence_ids).issubset(allowed_ids):
        raise ContractMismatchError("VLM response references unknown evidence IDs")
    if parsed.answer_type is not request.question.answer_type:
        raise ContractMismatchError("VLM response answer_type does not match the question")
    return parsed


__all__ = ["EVIDENCE_ONLY_INSTRUCTION", "build_vlm_request", "validate_vlm_response"]
