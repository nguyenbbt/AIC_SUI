"""Deterministic advanced-mode fakes and the shared Wave 2 fixture."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from online.domain.candidates import (
    CandidateDiagnostics,
    CandidateEvidence,
    FusedFrameCandidate,
)
from online.domain.enums import RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DataInfrastructureError,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.domain.trake import TRAKEEvent, TRAKEQuery
from online.domain.vqa import (
    ASREvidence,
    ImageEvidence,
    OCREvidence,
    SummaryEvidence,
    VLMConfidence,
    VLMRequest,
    VLMResponse,
    VLMResponseStatus,
    VQAAnswerType,
    VQAQuestion,
)
from online.ports.records import FrameMetadata
from online.ports.visual_corpus import OrderedVisualFrame

from .fakes import FakeMetadataReaderPort


@dataclass(frozen=True)
class AdvancedFakeBehavior:
    """One injected, shared-domain failure for an advanced fake operation."""

    error: DataInfrastructureError | None = None

    def __post_init__(self) -> None:
        if self.error is not None and not isinstance(
            self.error, DataInfrastructureError
        ):
            raise TypeError("error must be a DataInfrastructureError or None")

    def run(self) -> None:
        if self.error is not None:
            raise self.error


@dataclass(frozen=True)
class EncoderCall:
    text_count: int


@dataclass(frozen=True)
class VisualCorpusCall:
    operation: str
    video_id: str | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class EvidenceHydrationCall:
    operation: str
    record_ids: tuple[str, ...]
    start_sec: float | None = None
    end_sec: float | None = None


@dataclass(frozen=True)
class ImageResolverCall:
    frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class VLMCall:
    request_id: str
    evidence_ids: tuple[str, ...]


def _validated_ids(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise InvalidQueryError(f"{field_name} must be a sequence of strings")
    try:
        normalized = tuple(dict.fromkeys(values))
    except (TypeError, ValueError) as exc:
        raise InvalidQueryError(
            f"{field_name} must be a sequence of strings"
        ) from exc
    if any(
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        for value in normalized
    ):
        raise InvalidQueryError(
            f"{field_name} must contain non-empty strings without surrounding whitespace"
        )
    return normalized


def _validated_time_range(start_sec: float, end_sec: float) -> tuple[float, float]:
    values = (start_sec, end_sec)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise InvalidQueryError("ASR time range must contain finite numbers")
    normalized = (float(start_sec), float(end_sec))
    if normalized[0] < 0.0 or normalized[1] < normalized[0]:
        raise InvalidQueryError("ASR time range must satisfy 0 <= start_sec <= end_sec")
    return normalized


class FakeMappedTextEncoder:
    """TextEncoderPort fake with an explicit text-to-unit-vector mapping."""

    def __init__(self, vectors_by_text: Mapping[str, Sequence[float]]) -> None:
        if not isinstance(vectors_by_text, Mapping) or not vectors_by_text:
            raise ValueError("vectors_by_text must be a non-empty mapping")
        normalized: dict[str, tuple[float, ...]] = {}
        dimension: int | None = None
        for text, raw_vector in vectors_by_text.items():
            if not isinstance(text, str) or not text.strip() or text != text.strip():
                raise ValueError("encoder mapping keys must be non-empty normalized text")
            if isinstance(raw_vector, (str, bytes)):
                raise ValueError("encoder vectors must be numeric sequences")
            values = tuple(raw_vector)
            if (
                not values
                or any(isinstance(value, bool) for value in values)
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                raise ValueError("encoder vectors must be finite numeric sequences")
            vector = tuple(float(value) for value in values)
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("encoder vectors must have one shared dimension")
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("encoder vectors must be L2-normalized")
            normalized[text] = vector
        self._vectors_by_text = MappingProxyType(normalized)
        self._dimension = dimension or 0
        self._calls: list[EncoderCall] = []
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def calls(self) -> tuple[EncoderCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def encode_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        normalized = _validated_ids(texts, field_name="texts")
        # Duplicate event descriptions are a query-contract error, not something
        # this fake should silently collapse.
        if len(normalized) != len(tuple(texts)):
            raise InvalidQueryError("texts must not contain duplicates")
        with self._lock:
            self._calls.append(EncoderCall(text_count=len(normalized)))
        try:
            return tuple(self._vectors_by_text[text] for text in normalized)
        except KeyError as exc:
            raise ContractMismatchError(
                "text is not present in the deterministic encoder fixture",
                details={"text_count": len(normalized)},
            ) from exc


class FakeVisualCorpus:
    """Protocol-conformant, full ordered visual corpus fake."""

    def __init__(
        self,
        frames_by_video: Mapping[str, Sequence[OrderedVisualFrame]],
        *,
        list_behavior: AdvancedFakeBehavior | None = None,
        video_behaviors: Mapping[str, AdvancedFakeBehavior] | None = None,
    ) -> None:
        if not isinstance(frames_by_video, Mapping):
            raise TypeError("frames_by_video must be a mapping")
        normalized: dict[str, tuple[OrderedVisualFrame, ...]] = {}
        for video_id, frames in frames_by_video.items():
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError("video IDs must be non-empty strings")
            values = tuple(frames)
            if any(not isinstance(frame, OrderedVisualFrame) for frame in values):
                raise TypeError("visual corpus values must be OrderedVisualFrame values")
            if any(frame.video_id != video_id for frame in values):
                raise ContractMismatchError(
                    "visual fixture contains a frame under the wrong video"
                )
            normalized[video_id] = values
        self._frames_by_video = MappingProxyType(normalized)
        self._list_behavior = list_behavior or AdvancedFakeBehavior()
        self._video_behaviors = dict(video_behaviors or {})
        self._calls: list[VisualCorpusCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[VisualCorpusCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def list_video_ids(self) -> tuple[str, ...]:
        with self._lock:
            self._calls.append(VisualCorpusCall(operation="list_video_ids"))
        self._list_behavior.run()
        return tuple(sorted(self._frames_by_video))

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ):
        if (
            not isinstance(video_id, str)
            or not video_id.strip()
            or video_id != video_id.strip()
        ):
            raise InvalidQueryError("video_id must be a non-empty normalized string")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise InvalidQueryError("batch_size must be a positive integer")
        if video_id not in self._frames_by_video:
            raise InvalidQueryError("video_id is not present in the fake visual corpus")
        with self._lock:
            self._calls.append(
                VisualCorpusCall(
                    operation="iter_ordered_frame_embedding_batches",
                    video_id=video_id,
                    batch_size=batch_size,
                )
            )
        self._video_behaviors.get(video_id, AdvancedFakeBehavior()).run()
        frames = self._frames_by_video[video_id]
        return tuple(
            frames[offset : offset + batch_size]
            for offset in range(0, len(frames), batch_size)
        )


class FakeEvidenceHydrator:
    """Request-scoped OCR, ASR and summary evidence fake."""

    def __init__(
        self,
        *,
        ocr: Sequence[OCREvidence] = (),
        asr: Sequence[ASREvidence] = (),
        summaries: Sequence[SummaryEvidence] = (),
        behaviors: Mapping[str, AdvancedFakeBehavior] | None = None,
    ) -> None:
        self._ocr = tuple(ocr)
        self._asr = tuple(asr)
        self._summaries = tuple(summaries)
        self._behaviors = dict(behaviors or {})
        self._calls: list[EvidenceHydrationCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[EvidenceHydrationCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def _run(self, operation: str) -> None:
        self._behaviors.get(operation, AdvancedFakeBehavior()).run()

    def get_ocr_evidence(
        self, frame_ids: Sequence[str]
    ) -> tuple[OCREvidence, ...]:
        ids = _validated_ids(frame_ids, field_name="frame_ids")
        with self._lock:
            self._calls.append(EvidenceHydrationCall("ocr", ids))
        self._run("ocr")
        requested = set(ids)
        return tuple(record for record in self._ocr if record.frame_id in requested)

    def get_asr_evidence(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
    ) -> tuple[ASREvidence, ...]:
        ids = _validated_ids((video_id,), field_name="video_id")
        start, end = _validated_time_range(start_sec, end_sec)
        with self._lock:
            self._calls.append(
                EvidenceHydrationCall("asr", ids, start_sec=start, end_sec=end)
            )
        self._run("asr")
        return tuple(
            record
            for record in self._asr
            if record.video_id == video_id
            and record.end_time_sec >= start
            and record.start_time_sec <= end
        )

    def get_summary_evidence(
        self, video_ids: Sequence[str]
    ) -> tuple[SummaryEvidence, ...]:
        ids = _validated_ids(video_ids, field_name="video_ids")
        with self._lock:
            self._calls.append(EvidenceHydrationCall("summary", ids))
        self._run("summary")
        requested = set(ids)
        return tuple(
            record for record in self._summaries if record.video_id in requested
        )


class FakeImageResolver:
    """Safe fixture-reference image resolver with distinct missing/failure states."""

    def __init__(
        self,
        images: Mapping[str, ImageEvidence],
        *,
        behavior: AdvancedFakeBehavior | None = None,
    ) -> None:
        self._images = MappingProxyType(dict(images))
        self._behavior = behavior or AdvancedFakeBehavior()
        self._calls: list[ImageResolverCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[ImageResolverCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def resolve_images(
        self, frame_ids: Sequence[str]
    ) -> Mapping[str, ImageEvidence]:
        ids = _validated_ids(frame_ids, field_name="frame_ids")
        with self._lock:
            self._calls.append(ImageResolverCall(ids))
        self._behavior.run()
        return MappingProxyType(
            {frame_id: self._images[frame_id] for frame_id in ids if frame_id in self._images}
        )


class FakeVLMMode(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"


class FakeVLM:
    """Network-free VLM fake covering grounded and defensive-validation paths."""

    def __init__(
        self,
        mode: FakeVLMMode | str = FakeVLMMode.ANSWERED,
        *,
        answer: str = "Người đàn ông nâng một chiếc cốc.",
        grounded_evidence_ids: Sequence[str] | None = None,
    ) -> None:
        self.mode = FakeVLMMode(mode)
        self.answer_text = answer
        self.grounded_evidence_ids = (
            tuple(grounded_evidence_ids)
            if grounded_evidence_ids is not None
            else None
        )
        self._calls: list[VLMCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[VLMCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def answer(self, request: VLMRequest) -> Any:
        if not isinstance(request, VLMRequest):
            raise InvalidQueryError("request must be a validated VLMRequest")
        evidence_ids = tuple(item.evidence_id for item in request.evidence)
        with self._lock:
            self._calls.append(VLMCall(request.request_id, evidence_ids))
        if self.mode is FakeVLMMode.TIMEOUT:
            raise BranchTimeoutError("fake VLM timed out")
        if self.mode is FakeVLMMode.UNAVAILABLE:
            raise ResourceUnavailableError("fake VLM is unavailable")
        if self.mode is FakeVLMMode.MALFORMED:
            return {
                "status": "answered",
                "answer": self.answer_text,
                "evidence_ids": ("image:unknown",),
            }
        if self.mode is FakeVLMMode.INSUFFICIENT:
            return VLMResponse(
                status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                answer_type=request.question.answer_type,
                confidence=VLMConfidence.LOW,
                evidence_ids=(),
            )
        grounded_ids = (
            self.grounded_evidence_ids
            if self.grounded_evidence_ids is not None
            else evidence_ids[:1]
        )
        if not grounded_ids or not set(grounded_ids).issubset(evidence_ids):
            raise ContractMismatchError(
                "fake VLM grounded evidence IDs must be a non-empty request subset"
            )
        return VLMResponse(
            status=VLMResponseStatus.ANSWERED,
            answer=self.answer_text,
            answer_type=request.question.answer_type,
            confidence=VLMConfidence.HIGH,
            evidence_ids=grounded_ids,
        )


@dataclass(frozen=True)
class AdvancedModesFixture:
    """Single source of truth shared by TRAKE and VQA Wave 2 tests."""

    trake_query: TRAKEQuery
    event_vectors: Mapping[str, tuple[float, ...]]
    visual_frames_by_video: Mapping[str, tuple[OrderedVisualFrame, ...]]
    expected_dante_video_id: str
    expected_dante_positions: tuple[int, ...]
    expected_dante_score: float
    tie_video_id: str
    tied_sequence_positions: tuple[tuple[int, ...], ...]
    ranked_vqa_candidates: tuple[FusedFrameCandidate, ...]
    frame_metadata: tuple[FrameMetadata, ...]
    ocr_evidence: tuple[OCREvidence, ...]
    asr_evidence: tuple[ASREvidence, ...]
    summary_evidence: tuple[SummaryEvidence, ...]
    images_by_frame_id: Mapping[str, ImageEvidence]
    missing_image_frame_id: str
    vqa_question: VQAQuestion
    expected_vqa_selected_evidence_ids: tuple[str, ...]
    expected_vqa_answer_evidence_ids: tuple[str, ...]

    def text_encoder(self) -> FakeMappedTextEncoder:
        return FakeMappedTextEncoder(self.event_vectors)

    def visual_corpus(
        self,
        *,
        list_behavior: AdvancedFakeBehavior | None = None,
        video_behaviors: Mapping[str, AdvancedFakeBehavior] | None = None,
    ) -> FakeVisualCorpus:
        return FakeVisualCorpus(
            self.visual_frames_by_video,
            list_behavior=list_behavior,
            video_behaviors=video_behaviors,
        )

    def metadata(self) -> FakeMetadataReaderPort:
        return FakeMetadataReaderPort(self.frame_metadata)

    def evidence_hydrator(
        self,
        *,
        behaviors: Mapping[str, AdvancedFakeBehavior] | None = None,
    ) -> FakeEvidenceHydrator:
        return FakeEvidenceHydrator(
            ocr=self.ocr_evidence,
            asr=self.asr_evidence,
            summaries=self.summary_evidence,
            behaviors=behaviors,
        )

    def image_resolver(
        self, *, behavior: AdvancedFakeBehavior | None = None
    ) -> FakeImageResolver:
        return FakeImageResolver(self.images_by_frame_id, behavior=behavior)

    def vlm(
        self,
        mode: FakeVLMMode | str = FakeVLMMode.ANSWERED,
        *,
        grounded_evidence_ids: Sequence[str] | None = None,
    ) -> FakeVLM:
        if grounded_evidence_ids is None:
            grounded_ids = self.expected_vqa_answer_evidence_ids
        else:
            grounded_ids = _validated_ids(
                grounded_evidence_ids,
                field_name="grounded_evidence_ids",
            )
            raw_ids = tuple(grounded_evidence_ids)
            if len(grounded_ids) != len(raw_ids):
                raise InvalidQueryError(
                    "grounded_evidence_ids must not contain duplicates"
                )
        return FakeVLM(
            mode,
            grounded_evidence_ids=grounded_ids,
        )


_EVENT_TEXTS = (
    "Một người đàn ông bước vào phòng.",
    "Người đàn ông ngồi xuống ghế.",
    "Người đàn ông nâng một chiếc cốc.",
)
_EVENT_VECTORS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)
_DISTRACTOR = (0.0, 0.0, 0.0, 1.0)


def _visual_frame(
    video_id: str,
    local_index: int,
    vector: tuple[float, ...],
) -> OrderedVisualFrame:
    return OrderedVisualFrame(
        frame_id=f"{video_id}_{local_index:05d}_010",
        video_id=video_id,
        shot_id=local_index,
        local_index=local_index,
        timestamp_sec=float(local_index * 2),
        vector=vector,
    )


def _metadata(frame: OrderedVisualFrame) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame.frame_id,
        video_id=frame.video_id,
        shot_id=frame.shot_id,
        timestamp_sec=frame.timestamp_sec,
    )


def _fused(frame: OrderedVisualFrame, score: float) -> FusedFrameCandidate:
    return FusedFrameCandidate(
        frame_id=frame.frame_id,
        video_id=frame.video_id,
        shot_id=frame.shot_id,
        timestamp_sec=frame.timestamp_sec,
        final_score=score,
        branch_scores={RetrievalBranch.VISUAL_DENSE: score},
        evidence=(
            CandidateEvidence(
                branch=RetrievalBranch.VISUAL_DENSE,
                query_variant_id="q0",
                raw_score=score,
                normalized_score=score,
                backend="derived",
                source_resource="advanced_fixture",
                source_candidate_id=frame.frame_id,
            ),
        ),
        diagnostics=CandidateDiagnostics(),
    )


def build_advanced_modes_fixture() -> AdvancedModesFixture:
    """Build the deterministic four-video TRAKE/VQA Wave 2 fixture."""

    events = tuple(
        TRAKEEvent(event_id=f"event-{index}", text=text)
        for index, text in enumerate(_EVENT_TEXTS, start=1)
    )
    trake_query = TRAKEQuery(
        query_id="trake-wave2-fixture",
        events=events,
        top_k_videos=3,
    )
    frames = {
        "V001": tuple(
            _visual_frame("V001", index, vector)
            for index, vector in enumerate(
                (
                    _EVENT_VECTORS[0],
                    _DISTRACTOR,
                    _EVENT_VECTORS[1],
                    _DISTRACTOR,
                    _EVENT_VECTORS[2],
                    _DISTRACTOR,
                )
            )
        ),
        "V002": tuple(
            _visual_frame("V002", index, vector)
            for index, vector in enumerate(
                (
                    _EVENT_VECTORS[2],
                    _EVENT_VECTORS[2],
                    _EVENT_VECTORS[1],
                    _EVENT_VECTORS[1],
                    _EVENT_VECTORS[0],
                    _EVENT_VECTORS[0],
                )
            )
        ),
        "V003": tuple(
            _visual_frame("V003", index, vector)
            for index, vector in enumerate(
                (
                    (0.8, 0.0, 0.0, 0.6),
                    (0.8, 0.0, 0.0, 0.6),
                    (0.0, 0.8, 0.0, 0.6),
                    (0.0, 0.8, 0.0, 0.6),
                    (0.0, 0.0, 0.8, 0.6),
                    (0.0, 0.0, 0.8, 0.6),
                )
            )
        ),
        "V004": tuple(
            _visual_frame("V004", index, vector)
            for index, vector in enumerate(_EVENT_VECTORS[:2])
        ),
    }
    all_metadata = tuple(
        _metadata(frame)
        for video_id in sorted(frames)
        for frame in frames[video_id]
    )

    vqa_primary = frames["V001"][4]
    vqa_neighbor = frames["V001"][3]
    vqa_secondary = frames["V002"][2]
    ranked_candidates = (
        _fused(vqa_primary, 0.95),
        _fused(vqa_secondary, 0.85),
        _fused(frames["V001"][2], 0.75),
    )
    ocr = (
        OCREvidence(
            evidence_id=f"ocr:{vqa_primary.frame_id}",
            video_id=vqa_primary.video_id,
            frame_id=vqa_primary.frame_id,
            text="CỐC",
        ),
        OCREvidence(
            evidence_id=f"ocr:{vqa_secondary.frame_id}",
            video_id=vqa_secondary.video_id,
            frame_id=vqa_secondary.frame_id,
            text="PHÒNG KHÁCH",
        ),
    )
    asr = (
        ASREvidence(
            evidence_id="asr:V001:interval-1",
            video_id="V001",
            interval_id="interval-1",
            start_time_sec=6.0,
            end_time_sec=10.0,
            text="Anh ấy nâng chiếc cốc lên.",
        ),
        ASREvidence(
            evidence_id="asr:V002:interval-1",
            video_id="V002",
            interval_id="interval-1",
            start_time_sec=3.0,
            end_time_sec=6.0,
            text="Một cảnh khác trong phòng.",
        ),
    )
    summaries = (
        SummaryEvidence(
            evidence_id="summary:V001",
            video_id="V001",
            text="Một người bước vào, ngồi xuống rồi nâng một chiếc cốc.",
        ),
        SummaryEvidence(
            evidence_id="summary:V002",
            video_id="V002",
            text="Các hành động tương tự xuất hiện theo thứ tự ngược.",
        ),
    )
    image_frames = (vqa_primary, vqa_neighbor, vqa_secondary)
    images = {
        frame.frame_id: ImageEvidence(
            evidence_id=f"image:{frame.frame_id}",
            video_id=frame.video_id,
            frame_id=frame.frame_id,
            shot_id=frame.shot_id,
            timestamp_sec=frame.timestamp_sec,
            image_reference=f"fixture://advanced-images/{frame.frame_id}",
        )
        for frame in image_frames
    }
    expected_selected_evidence_ids = (
        f"image:{vqa_primary.frame_id}",
        f"image:{vqa_secondary.frame_id}",
        f"image:{vqa_neighbor.frame_id}",
        f"ocr:{vqa_primary.frame_id}",
        f"ocr:{vqa_secondary.frame_id}",
        "asr:V001:interval-1",
        "asr:V002:interval-1",
        "summary:V001",
        "summary:V002",
    )
    expected_answer_evidence_ids = (
        f"image:{vqa_primary.frame_id}",
        f"ocr:{vqa_primary.frame_id}",
        "asr:V001:interval-1",
        "summary:V001",
    )

    return AdvancedModesFixture(
        trake_query=trake_query,
        event_vectors=MappingProxyType(dict(zip(_EVENT_TEXTS, _EVENT_VECTORS))),
        visual_frames_by_video=MappingProxyType(frames),
        expected_dante_video_id="V001",
        expected_dante_positions=(0, 2, 4),
        expected_dante_score=2.996,
        tie_video_id="V003",
        tied_sequence_positions=((1, 2, 4), (1, 3, 4)),
        ranked_vqa_candidates=ranked_candidates,
        frame_metadata=all_metadata,
        ocr_evidence=ocr,
        asr_evidence=asr,
        summary_evidence=summaries,
        images_by_frame_id=MappingProxyType(images),
        missing_image_frame_id=frames["V001"][5].frame_id,
        vqa_question=VQAQuestion(
            question_id="vqa-wave2-fixture",
            question="Người đàn ông làm gì sau khi ngồi xuống?",
            answer_type=VQAAnswerType.SHORT_TEXT,
        ),
        expected_vqa_selected_evidence_ids=expected_selected_evidence_ids,
        expected_vqa_answer_evidence_ids=expected_answer_evidence_ids,
    )


__all__ = [
    "AdvancedFakeBehavior",
    "AdvancedModesFixture",
    "EncoderCall",
    "EvidenceHydrationCall",
    "FakeEvidenceHydrator",
    "FakeImageResolver",
    "FakeMappedTextEncoder",
    "FakeVLM",
    "FakeVLMMode",
    "FakeVisualCorpus",
    "ImageResolverCall",
    "VLMCall",
    "VisualCorpusCall",
    "build_advanced_modes_fixture",
]
