"""Public, SDK-neutral VQA evidence and VLM contracts."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import AfterValidator, Field, field_validator, model_validator

from .base import FiniteFloat, NonEmptyStr, StrictFrozenModel, StrictIntValue
from .errors import ContractMismatchError
from .identifiers import validate_canonical_frame_id


class VQAAnswerType(str, Enum):
    SHORT_TEXT = "short_text"
    YES_NO = "yes_no"
    NUMBER = "number"
    LIST = "list"


class EvidenceType(str, Enum):
    IMAGE = "image"
    OCR = "ocr"
    ASR = "asr"
    SUMMARY = "summary"


class VLMResponseStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class VLMConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _safe_opaque_reference(value: str) -> str:
    if not value.strip():
        raise ValueError("reference must not be empty or whitespace")
    if value != value.strip():
        raise ValueError("reference must not contain surrounding whitespace")
    if "\x00" in value:
        raise ValueError("reference must not contain NUL")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or bool(windows.drive):
        raise ValueError("local absolute paths are not allowed in public evidence")
    parsed = urlsplit(value)
    if parsed.scheme.lower() == "file":
        raise ValueError("file:// references are not allowed in public evidence")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials are not allowed in public evidence references")
    sensitive_query_parts = ("api_key", "apikey", "token", "secret", "signature", "credential")
    if any(
        any(part in key.lower() for part in sensitive_query_parts)
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    ):
        raise ValueError("secret-bearing query parameters are not allowed in public evidence")
    return value


EvidenceId = Annotated[str, AfterValidator(_safe_opaque_reference)]
ImageReference = Annotated[str, AfterValidator(_safe_opaque_reference)]


class VQAQuestion(StrictFrozenModel):
    question_id: NonEmptyStr
    question: NonEmptyStr
    answer_type: VQAAnswerType


class VQAEvidenceBudget(StrictFrozenModel):
    """Frozen DD-030 defaults, configurable only through validated values."""

    max_videos: StrictIntValue = Field(default=3, ge=1)
    max_primary_frames_per_video: StrictIntValue = Field(default=3, ge=1)
    max_primary_frames_total: StrictIntValue = Field(default=8, ge=1)
    max_images_total: StrictIntValue = Field(default=12, ge=1)
    max_ocr_chars: StrictIntValue = Field(default=2_000, ge=1)
    max_asr_chars: StrictIntValue = Field(default=4_000, ge=1)
    max_summary_chars_per_video: StrictIntValue = Field(default=800, ge=1)
    max_summary_chars_total: StrictIntValue = Field(default=2_400, ge=1)
    max_text_chars_total: StrictIntValue = Field(default=8_000, ge=1)
    asr_window_sec: Annotated[FiniteFloat, Field(gt=0.0)] = 5.0

    @model_validator(mode="after")
    def validate_consistent_caps(self) -> "VQAEvidenceBudget":
        if self.max_videos > self.max_primary_frames_total:
            raise ValueError("max_videos cannot exceed max_primary_frames_total")
        if self.max_primary_frames_per_video > self.max_primary_frames_total:
            raise ValueError(
                "max_primary_frames_per_video cannot exceed max_primary_frames_total"
            )
        if self.max_primary_frames_total > self.max_images_total:
            raise ValueError("max_primary_frames_total cannot exceed max_images_total")
        if self.max_ocr_chars > self.max_text_chars_total:
            raise ValueError("max_ocr_chars cannot exceed max_text_chars_total")
        if self.max_asr_chars > self.max_text_chars_total:
            raise ValueError("max_asr_chars cannot exceed max_text_chars_total")
        if self.max_summary_chars_per_video > self.max_summary_chars_total:
            raise ValueError(
                "max_summary_chars_per_video cannot exceed max_summary_chars_total"
            )
        if self.max_summary_chars_total > self.max_text_chars_total:
            raise ValueError("max_summary_chars_total cannot exceed max_text_chars_total")
        return self


class EvidenceReference(StrictFrozenModel):
    """Stable identity and candidate level shared by every evidence item."""

    evidence_id: EvidenceId
    evidence_type: EvidenceType
    video_id: NonEmptyStr


class ImageEvidence(EvidenceReference):
    evidence_type: Literal[EvidenceType.IMAGE] = EvidenceType.IMAGE
    frame_id: NonEmptyStr
    keyframe_no: StrictIntValue = Field(ge=1)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    source_frame_idx: StrictIntValue = Field(ge=0)
    image_reference: ImageReference

    @model_validator(mode="after")
    def validate_frame_identity(self) -> "ImageEvidence":
        _validate_frame_identity(
            self.frame_id,
            self.video_id,
            keyframe_no=self.keyframe_no,
        )
        return self


class OCREvidence(EvidenceReference):
    evidence_type: Literal[EvidenceType.OCR] = EvidenceType.OCR
    frame_id: NonEmptyStr
    text: NonEmptyStr

    @model_validator(mode="after")
    def validate_frame_identity(self) -> "OCREvidence":
        _validate_frame_identity(self.frame_id, self.video_id)
        return self


class ASREvidence(EvidenceReference):
    evidence_type: Literal[EvidenceType.ASR] = EvidenceType.ASR
    interval_id: NonEmptyStr
    start_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    end_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    text: NonEmptyStr

    @model_validator(mode="after")
    def validate_interval(self) -> "ASREvidence":
        if self.end_time_sec < self.start_time_sec:
            raise ValueError("end_time_sec must be >= start_time_sec")
        return self


class SummaryEvidence(EvidenceReference):
    evidence_type: Literal[EvidenceType.SUMMARY] = EvidenceType.SUMMARY
    text: NonEmptyStr


VQAEvidence = ImageEvidence | OCREvidence | ASREvidence | SummaryEvidence


class VLMRequest(StrictFrozenModel):
    """Validated evidence-only request passed to a mockable VLM port."""

    request_id: NonEmptyStr
    question: VQAQuestion
    evidence: tuple[VQAEvidence, ...] = Field(min_length=1)
    temperature: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.1
    max_output_tokens: StrictIntValue = Field(default=512, ge=1)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "VLMRequest":
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("VLM request evidence IDs must be unique")
        return self


class VLMResponse(StrictFrozenModel):
    status: VLMResponseStatus
    answer: str | None = None
    answer_type: VQAAnswerType
    confidence: VLMConfidence
    evidence_ids: tuple[EvidenceId, ...] = ()

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_empty_answer(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def validate_status_contract(self) -> "VLMResponse":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("VLM response evidence IDs must be unique")
        if self.status is VLMResponseStatus.ANSWERED:
            if self.answer is None:
                raise ValueError("answered response requires a non-empty answer")
            if not self.evidence_ids:
                raise ValueError("answered response requires at least one evidence ID")
        elif self.answer is not None:
            raise ValueError("insufficient_evidence response must not contain an answer")
        return self


class VQADiagnostics(StrictFrozenModel):
    retrieved_frame_count: StrictIntValue = Field(default=0, ge=0)
    selected_image_count: StrictIntValue = Field(default=0, ge=0)
    selected_text_evidence_count: StrictIntValue = Field(default=0, ge=0)
    dropped_evidence_count: StrictIntValue = Field(default=0, ge=0)
    missing_evidence_count: StrictIntValue = Field(default=0, ge=0)
    vlm_latency_ms: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    vlm_retry_count: StrictIntValue = Field(default=0, ge=0)
    warnings: tuple[NonEmptyStr, ...] = ()


class VQAResult(StrictFrozenModel):
    question_id: NonEmptyStr
    response: VLMResponse
    evidence: tuple[VQAEvidence, ...] = ()
    diagnostics: VQADiagnostics = Field(default_factory=VQADiagnostics)

    @model_validator(mode="after")
    def validate_grounding(self) -> "VQAResult":
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("VQA result evidence IDs must be unique")
        unknown_ids = set(self.response.evidence_ids).difference(evidence_ids)
        if unknown_ids:
            raise ValueError("response evidence IDs must be a subset of result evidence")
        return self


def _validate_frame_identity(
    frame_id: str,
    video_id: str,
    keyframe_no: int | None = None,
) -> None:
    try:
        validate_canonical_frame_id(
            frame_id,
            video_id=video_id,
            keyframe_no=keyframe_no,
        )
    except ContractMismatchError as exc:
        raise ValueError(exc.message) from exc
