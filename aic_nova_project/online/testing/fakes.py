"""Deterministic, behavior-conformant fakes shared by B/C tests."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from online.domain.candidates import ObjectDetection
from online.domain.enums import RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DataInfrastructureError,
    DimensionMismatchError,
    InvalidQueryError,
)
from online.ports.records import (
    ASRSearchHit,
    FrameMetadata,
    FrameSearchHit,
    VideoSearchHit,
)


@dataclass(frozen=True)
class FakeBranchBehavior:
    """Injectable deterministic branch behavior for timeout/failure tests."""

    error: DataInfrastructureError | None = None
    delay_sec: float = 0.0
    entered: threading.Event | None = None
    release: threading.Event | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.delay_sec) or self.delay_sec < 0:
            raise ValueError("delay_sec must be finite and >= 0")


@dataclass(frozen=True)
class MilvusCall:
    branch: RetrievalBranch
    vector: tuple[float, ...]
    top_k: int


@dataclass(frozen=True)
class ElasticsearchCall:
    branch: RetrievalBranch
    query: str
    top_k: int
    fuzzy: bool | None


@dataclass(frozen=True)
class MetadataCall:
    frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObjectCall:
    frame_ids: tuple[str, ...]
    label: str | None
    min_confidence: float


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise InvalidQueryError("top_k must be >= 1")


def _validate_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int | None,
    require_unit_norm: bool,
    norm_tolerance: float,
) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise InvalidQueryError("query vector must be a numeric sequence")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise InvalidQueryError("query vector must contain numeric values") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise InvalidQueryError("query vector must be non-empty and finite")
    if expected_dimension is not None and len(values) != expected_dimension:
        raise DimensionMismatchError(
            "fake query vector dimension does not match configured dimension",
            details={"expected": expected_dimension, "actual": len(values)},
        )
    if require_unit_norm:
        norm = math.sqrt(sum(value * value for value in values))
        if abs(norm - 1.0) > norm_tolerance:
            raise InvalidQueryError("query vector must be L2-normalized")
    return values


def _run_behavior(behavior: FakeBranchBehavior | None) -> None:
    if behavior is None:
        return
    if behavior.entered is not None:
        behavior.entered.set()
    if behavior.release is not None:
        if not behavior.release.wait(timeout=2.0):
            raise BranchTimeoutError("fake branch release event timed out")
    elif behavior.delay_sec:
        time.sleep(min(behavior.delay_sec, 0.05))
    if behavior.error is not None:
        raise behavior.error


class FakeTextEncoder:
    """Deterministic, normalized TextEncoderPort fake for B/C integration tests."""

    def __init__(self, dimension: int = 4) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self._dimension = dimension
        self.calls: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence")
        values = tuple(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("texts must contain non-empty strings")
        self.calls.append(values)

        vectors: list[tuple[float, ...]] = []
        for text in values:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = tuple(
                (digest[index % len(digest)] - 127.5) / 127.5
                for index in range(self._dimension)
            )
            norm = math.sqrt(sum(value * value for value in raw))
            vectors.append(tuple(value / norm for value in raw))
        return tuple(vectors)


class FakeMilvusSearchPort:
    def __init__(
        self,
        *,
        visual: Sequence[FrameSearchHit] = (),
        ocr: Sequence[FrameSearchHit] = (),
        asr: Sequence[ASRSearchHit] = (),
        summary: Sequence[VideoSearchHit] = (),
        expected_dimensions: Mapping[RetrievalBranch, int] | None = None,
        require_unit_norm: bool = True,
        norm_tolerance: float = 1e-3,
        behaviors: Mapping[RetrievalBranch, FakeBranchBehavior] | None = None,
    ) -> None:
        self.visual = tuple(visual)
        self.ocr = tuple(ocr)
        self.asr = tuple(asr)
        self.summary = tuple(summary)
        self.expected_dimensions = dict(expected_dimensions or {})
        self.require_unit_norm = require_unit_norm
        self.norm_tolerance = norm_tolerance
        self.behaviors = dict(behaviors or {})
        self._calls: list[MilvusCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[MilvusCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def _compatibility_calls(
        self,
        branch: RetrievalBranch,
    ) -> list[tuple[tuple[float, ...], int]]:
        return [
            (call.vector, call.top_k)
            for call in self.calls
            if call.branch is branch
        ]

    @property
    def visual_calls(self) -> list[tuple[tuple[float, ...], int]]:
        return self._compatibility_calls(RetrievalBranch.VISUAL_DENSE)

    @property
    def ocr_calls(self) -> list[tuple[tuple[float, ...], int]]:
        return self._compatibility_calls(RetrievalBranch.OCR_DENSE)

    @property
    def asr_calls(self) -> list[tuple[tuple[float, ...], int]]:
        return self._compatibility_calls(RetrievalBranch.ASR_DENSE)

    @property
    def summary_calls(self) -> list[tuple[tuple[float, ...], int]]:
        return self._compatibility_calls(RetrievalBranch.SUMMARY_DENSE)

    def _take(
        self,
        branch: RetrievalBranch,
        values: Sequence[object],
        vector: Sequence[float],
        top_k: int,
    ) -> Sequence[object]:
        _validate_top_k(top_k)
        validated = _validate_vector(
            vector,
            expected_dimension=self.expected_dimensions.get(branch),
            require_unit_norm=self.require_unit_norm,
            norm_tolerance=self.norm_tolerance,
        )
        with self._lock:
            self._calls.append(MilvusCall(branch, validated, top_k))
        _run_behavior(self.behaviors.get(branch))
        return tuple(values[:top_k])

    def search_visual(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]:
        return self._take(RetrievalBranch.VISUAL_DENSE, self.visual, vector, top_k)  # type: ignore[return-value]

    def search_ocr(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]:
        return self._take(RetrievalBranch.OCR_DENSE, self.ocr, vector, top_k)  # type: ignore[return-value]

    def search_asr(self, vector: Sequence[float], top_k: int) -> Sequence[ASRSearchHit]:
        return self._take(RetrievalBranch.ASR_DENSE, self.asr, vector, top_k)  # type: ignore[return-value]

    def search_summary(self, vector: Sequence[float], top_k: int) -> Sequence[VideoSearchHit]:
        return self._take(RetrievalBranch.SUMMARY_DENSE, self.summary, vector, top_k)  # type: ignore[return-value]


class FakeElasticsearchSearchPort:
    def __init__(
        self,
        *,
        ocr: Sequence[FrameSearchHit] = (),
        asr: Sequence[ASRSearchHit] = (),
        summary: Sequence[VideoSearchHit] = (),
        behaviors: Mapping[RetrievalBranch, FakeBranchBehavior] | None = None,
    ) -> None:
        self.ocr = tuple(ocr)
        self.asr = tuple(asr)
        self.summary = tuple(summary)
        self.behaviors = dict(behaviors or {})
        self._calls: list[ElasticsearchCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[ElasticsearchCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def _compatibility_calls(
        self,
        branch: RetrievalBranch,
    ) -> list[tuple[str, int, bool | None]]:
        return [
            (call.query, call.top_k, call.fuzzy)
            for call in self.calls
            if call.branch is branch
        ]

    @property
    def ocr_calls(self) -> list[tuple[str, int, bool | None]]:
        return self._compatibility_calls(RetrievalBranch.OCR_BM25)

    @property
    def asr_calls(self) -> list[tuple[str, int, bool | None]]:
        return self._compatibility_calls(RetrievalBranch.ASR_BM25)

    @property
    def summary_calls(self) -> list[tuple[str, int, bool | None]]:
        return self._compatibility_calls(RetrievalBranch.SUMMARY_BM25)

    def _validate(
        self,
        branch: RetrievalBranch,
        query: str,
        top_k: int,
        fuzzy: bool | None,
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("lexical query must not be empty")
        _validate_top_k(top_k)
        if fuzzy is not None and not isinstance(fuzzy, bool):
            raise InvalidQueryError("fuzzy must be a boolean or None")
        with self._lock:
            self._calls.append(ElasticsearchCall(branch, query, top_k, fuzzy))
        _run_behavior(self.behaviors.get(branch))

    def search_ocr(
        self, query: str, top_k: int, *, fuzzy: bool | None = None
    ) -> Sequence[FrameSearchHit]:
        self._validate(RetrievalBranch.OCR_BM25, query, top_k, fuzzy)
        return self.ocr[:top_k]

    def search_asr(
        self, query: str, top_k: int, *, fuzzy: bool | None = None
    ) -> Sequence[ASRSearchHit]:
        self._validate(RetrievalBranch.ASR_BM25, query, top_k, fuzzy)
        return self.asr[:top_k]

    def search_summary(
        self, query: str, top_k: int, *, fuzzy: bool | None = None
    ) -> Sequence[VideoSearchHit]:
        self._validate(RetrievalBranch.SUMMARY_BM25, query, top_k, fuzzy)
        return self.summary[:top_k]


class FakeMetadataReaderPort:
    def __init__(self, frames: Sequence[FrameMetadata]) -> None:
        if len({frame.frame_id for frame in frames}) != len(frames):
            raise ContractMismatchError("duplicate frame IDs in fake metadata")
        self._frames = {frame.frame_id: frame for frame in frames}
        self._calls: list[MetadataCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[MetadataCall, ...]:
        with self._lock:
            return tuple(self._calls)

    @staticmethod
    def _validate_ids(frame_ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(dict.fromkeys(frame_ids))
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise InvalidQueryError("frame_ids must contain non-empty strings")
        return ids

    def get_frames_by_ids(self, frame_ids: Sequence[str]) -> Mapping[str, FrameMetadata]:
        ids = self._validate_ids(frame_ids)
        with self._lock:
            self._calls.append(MetadataCall(ids))
        return {
            frame_id: self._frames[frame_id]
            for frame_id in ids
            if frame_id in self._frames
        }

    def get_ordered_frames_by_video(self, video_id: str) -> Sequence[FrameMetadata]:
        if not isinstance(video_id, str) or not video_id.strip():
            raise InvalidQueryError("video_id must not be empty")
        return tuple(
            sorted(
                (frame for frame in self._frames.values() if frame.video_id == video_id),
                key=lambda frame: (frame.timestamp_sec, frame.frame_id),
            )
        )


class FakeObjectReaderPort:
    def __init__(self, objects: Mapping[str, Sequence[ObjectDetection]]) -> None:
        self._objects = {frame_id: tuple(values) for frame_id, values in objects.items()}
        self._calls: list[ObjectCall] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[ObjectCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def get_objects_by_frame_ids(
        self,
        frame_ids: Sequence[str],
        *,
        label: str | None = None,
        min_confidence: float = 0.0,
    ) -> Mapping[str, Sequence[ObjectDetection]]:
        ids = tuple(dict.fromkeys(frame_ids))
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise InvalidQueryError("frame_ids must contain non-empty strings")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise InvalidQueryError("label must not be empty")
        if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
            raise InvalidQueryError("min_confidence must be within [0, 1]")
        with self._lock:
            self._calls.append(ObjectCall(ids, label, min_confidence))
        output: dict[str, tuple[ObjectDetection, ...]] = {}
        for frame_id in ids:
            values = self._objects.get(frame_id, ())
            output[frame_id] = tuple(
                value
                for value in values
                if value.confidence >= min_confidence
                and (label is None or value.label == label)
            )
        return output


@dataclass(frozen=True)
class IntegrationFixture:
    schema_version: str
    frames: tuple[FrameMetadata, ...]
    objects: Mapping[str, tuple[ObjectDetection, ...]]
    visual_hits: tuple[FrameSearchHit, ...]
    ocr_dense_hits: tuple[FrameSearchHit, ...]
    ocr_lexical_hits: tuple[FrameSearchHit, ...]
    asr_dense_hits: tuple[ASRSearchHit, ...]
    asr_lexical_hits: tuple[ASRSearchHit, ...]
    summary_dense_hits: tuple[VideoSearchHit, ...]
    summary_lexical_hits: tuple[VideoSearchHit, ...]
    missing_metadata_hit: FrameSearchHit
    metadata_mismatch_hit: FrameSearchHit

    # Compatibility aliases retained for the first merged fixture contract.
    @property
    def ocr_hits(self) -> tuple[FrameSearchHit, ...]:
        return self.ocr_dense_hits

    @property
    def asr_hits(self) -> tuple[ASRSearchHit, ...]:
        return self.asr_dense_hits

    @property
    def summary_hits(self) -> tuple[VideoSearchHit, ...]:
        return self.summary_dense_hits

    def milvus(self, *, behaviors: Mapping[RetrievalBranch, FakeBranchBehavior] | None = None) -> FakeMilvusSearchPort:
        return FakeMilvusSearchPort(
            visual=self.visual_hits + (self.missing_metadata_hit, self.metadata_mismatch_hit),
            ocr=self.ocr_dense_hits,
            asr=self.asr_dense_hits,
            summary=self.summary_dense_hits,
            behaviors=behaviors,
        )

    def elasticsearch(self, *, behaviors: Mapping[RetrievalBranch, FakeBranchBehavior] | None = None) -> FakeElasticsearchSearchPort:
        return FakeElasticsearchSearchPort(
            ocr=self.ocr_lexical_hits,
            asr=self.asr_lexical_hits,
            summary=self.summary_lexical_hits,
            behaviors=behaviors,
        )

    def metadata(self) -> FakeMetadataReaderPort:
        return FakeMetadataReaderPort(self.frames)

    def object_reader(self) -> FakeObjectReaderPort:
        return FakeObjectReaderPort(self.objects)


def build_integration_fixture() -> IntegrationFixture:
    """Two-video fixture with all seven branch levels and ASR edge cases."""

    frames = (
        FrameMetadata(frame_id="V001_00000_015", video_id="V001", shot_id=0, timestamp_sec=1.5),
        FrameMetadata(frame_id="V001_00000_050", video_id="V001", shot_id=0, timestamp_sec=5.0),
        FrameMetadata(frame_id="V001_00001_050", video_id="V001", shot_id=1, timestamp_sec=10.0),
        FrameMetadata(frame_id="V002_00000_025", video_id="V002", shot_id=0, timestamp_sec=2.5),
        FrameMetadata(frame_id="V002_00001_075", video_id="V002", shot_id=1, timestamp_sec=17.5),
    )
    objects = {
        "V001_00000_015": (),
        "V001_00000_050": (
            ObjectDetection(
                label="person",
                confidence=0.95,
                x_min=10,
                y_min=20,
                x_max=110,
                y_max=220,
                model_source="yolo_world",
            ),
        ),
        "V001_00001_050": (
            ObjectDetection(
                label="person",
                confidence=0.91,
                x_min=15,
                y_min=25,
                x_max=115,
                y_max=225,
                model_source="yolo_world",
            ),
            ObjectDetection(
                label="person",
                confidence=0.35,
                x_min=120,
                y_min=30,
                x_max=200,
                y_max=230,
                model_source="co_detr",
            ),
            ObjectDetection(
                label="car",
                confidence=0.88,
                x_min=210,
                y_min=120,
                x_max=410,
                y_max=300,
                model_source="co_detr",
            ),
        ),
        "V002_00000_025": (
            ObjectDetection(
                label="bicycle",
                confidence=0.82,
                x_min=5,
                y_min=40,
                x_max=90,
                y_max=180,
            ),
        ),
    }
    visual_hits = tuple(
        FrameSearchHit(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            shot_id=frame.shot_id,
            raw_score=0.95 - index * 0.05,
        )
        for index, frame in enumerate(frames)
    )
    ocr_dense_hits = (
        FrameSearchHit(
            frame_id="V001_00000_050",
            video_id="V001",
            shot_id=None,
            raw_score=0.84,
        ),
        FrameSearchHit(
            frame_id="V002_00001_075",
            video_id="V002",
            shot_id=None,
            raw_score=0.61,
        ),
    )
    ocr_lexical_hits = (
        FrameSearchHit(
            frame_id="V001_00000_050",
            video_id="V001",
            shot_id=0,
            raw_score=7.2,
        ),
        FrameSearchHit(
            frame_id="V002_00001_075",
            video_id="V002",
            shot_id=1,
            raw_score=5.1,
        ),
    )
    asr_dense_hits = (
        ASRSearchHit(
            video_id="V001",
            interval_id="overlap",
            start_time_sec=1.0,
            end_time_sec=6.0,
            raw_score=0.9,
            text="overlap case",
        ),
        ASRSearchHit(
            video_id="V001",
            interval_id="boundary",
            start_time_sec=10.0,
            end_time_sec=12.0,
            raw_score=0.8,
            text="boundary case",
        ),
        ASRSearchHit(
            video_id="V002",
            interval_id="no_overlap",
            start_time_sec=8.0,
            end_time_sec=9.0,
            raw_score=0.7,
            text="no overlap case",
        ),
    )
    asr_lexical_hits = tuple(asr_dense_hits)
    summary_dense_hits = (
        VideoSearchHit(video_id="V001", raw_score=0.85, summary="First video"),
        VideoSearchHit(video_id="V002", raw_score=0.75, summary="Second video"),
    )
    summary_lexical_hits = tuple(summary_dense_hits)
    return IntegrationFixture(
        schema_version="person-a-online-fixture-v2",
        frames=frames,
        objects=objects,
        visual_hits=visual_hits,
        ocr_dense_hits=ocr_dense_hits,
        ocr_lexical_hits=ocr_lexical_hits,
        asr_dense_hits=asr_dense_hits,
        asr_lexical_hits=asr_lexical_hits,
        summary_dense_hits=summary_dense_hits,
        summary_lexical_hits=summary_lexical_hits,
        missing_metadata_hit=FrameSearchHit(
            frame_id="V999_99999_999", video_id="V999", shot_id=99999, raw_score=0.1
        ),
        metadata_mismatch_hit=FrameSearchHit(
            frame_id="V001_00000_015", video_id="V999", shot_id=0, raw_score=0.05
        ),
    )
