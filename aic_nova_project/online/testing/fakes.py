"""Deterministic Protocol-conformant fakes shared by B/C tests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from online.domain.candidates import ObjectDetection
from online.ports.records import ASRSearchHit, FrameMetadata, FrameSearchHit, VideoSearchHit


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
    ) -> None:
        self.visual = tuple(visual)
        self.ocr = tuple(ocr)
        self.asr = tuple(asr)
        self.summary = tuple(summary)
        self.visual_calls: list[tuple[tuple[float, ...], int]] = []
        self.ocr_calls: list[tuple[tuple[float, ...], int]] = []
        self.asr_calls: list[tuple[tuple[float, ...], int]] = []
        self.summary_calls: list[tuple[tuple[float, ...], int]] = []

    @staticmethod
    def _take(values: Sequence[object], top_k: int) -> Sequence[object]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return tuple(values[:top_k])

    def search_visual(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]:
        self.visual_calls.append((tuple(float(value) for value in vector), top_k))
        return self._take(self.visual, top_k)  # type: ignore[return-value]

    def search_ocr(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]:
        self.ocr_calls.append((tuple(float(value) for value in vector), top_k))
        return self._take(self.ocr, top_k)  # type: ignore[return-value]

    def search_asr(self, vector: Sequence[float], top_k: int) -> Sequence[ASRSearchHit]:
        self.asr_calls.append((tuple(float(value) for value in vector), top_k))
        return self._take(self.asr, top_k)  # type: ignore[return-value]

    def search_summary(self, vector: Sequence[float], top_k: int) -> Sequence[VideoSearchHit]:
        self.summary_calls.append((tuple(float(value) for value in vector), top_k))
        return self._take(self.summary, top_k)  # type: ignore[return-value]


class FakeElasticsearchSearchPort:
    def __init__(
        self,
        *,
        ocr: Sequence[FrameSearchHit] = (),
        asr: Sequence[ASRSearchHit] = (),
        summary: Sequence[VideoSearchHit] = (),
    ) -> None:
        self.ocr = tuple(ocr)
        self.asr = tuple(asr)
        self.summary = tuple(summary)
        self.ocr_calls: list[tuple[str, int, bool | None]] = []
        self.asr_calls: list[tuple[str, int, bool | None]] = []
        self.summary_calls: list[tuple[str, int, bool | None]] = []

    @staticmethod
    def _validate(query: str, top_k: int) -> None:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

    def search_ocr(self, query: str, top_k: int, *, fuzzy: bool | None = None) -> Sequence[FrameSearchHit]:
        self._validate(query, top_k)
        self.ocr_calls.append((query, top_k, fuzzy))
        return self.ocr[:top_k]

    def search_asr(self, query: str, top_k: int, *, fuzzy: bool | None = None) -> Sequence[ASRSearchHit]:
        self._validate(query, top_k)
        self.asr_calls.append((query, top_k, fuzzy))
        return self.asr[:top_k]

    def search_summary(self, query: str, top_k: int, *, fuzzy: bool | None = None) -> Sequence[VideoSearchHit]:
        self._validate(query, top_k)
        self.summary_calls.append((query, top_k, fuzzy))
        return self.summary[:top_k]


class FakeMetadataReaderPort:
    def __init__(self, frames: Sequence[FrameMetadata]) -> None:
        self._frames = {frame.frame_id: frame for frame in frames}

    def get_frames_by_ids(self, frame_ids: Sequence[str]) -> Mapping[str, FrameMetadata]:
        return {frame_id: self._frames[frame_id] for frame_id in dict.fromkeys(frame_ids) if frame_id in self._frames}

    def get_ordered_frames_by_video(self, video_id: str) -> Sequence[FrameMetadata]:
        return tuple(
            sorted(
                (frame for frame in self._frames.values() if frame.video_id == video_id),
                key=lambda frame: (frame.timestamp_sec, frame.frame_id),
            )
        )


class FakeObjectReaderPort:
    def __init__(self, objects: Mapping[str, Sequence[ObjectDetection]]) -> None:
        self._objects = {frame_id: tuple(values) for frame_id, values in objects.items()}

    def get_objects_by_frame_ids(
        self,
        frame_ids: Sequence[str],
        *,
        label: str | None = None,
        min_confidence: float = 0.0,
    ) -> Mapping[str, Sequence[ObjectDetection]]:
        output: dict[str, tuple[ObjectDetection, ...]] = {}
        for frame_id in dict.fromkeys(frame_ids):
            values = self._objects.get(frame_id, ())
            output[frame_id] = tuple(
                value
                for value in values
                if value.confidence >= min_confidence and (label is None or value.label == label)
            )
        return output


@dataclass(frozen=True)
class IntegrationFixture:
    frames: tuple[FrameMetadata, ...]
    objects: Mapping[str, tuple[ObjectDetection, ...]]
    visual_hits: tuple[FrameSearchHit, ...]
    ocr_hits: tuple[FrameSearchHit, ...]
    asr_hits: tuple[ASRSearchHit, ...]
    summary_hits: tuple[VideoSearchHit, ...]
    missing_metadata_hit: FrameSearchHit

    def milvus(self) -> FakeMilvusSearchPort:
        return FakeMilvusSearchPort(
            visual=self.visual_hits + (self.missing_metadata_hit,),
            ocr=self.ocr_hits,
            asr=self.asr_hits,
            summary=self.summary_hits,
        )

    def metadata(self) -> FakeMetadataReaderPort:
        return FakeMetadataReaderPort(self.frames)

    def object_reader(self) -> FakeObjectReaderPort:
        return FakeObjectReaderPort(self.objects)


def build_integration_fixture() -> IntegrationFixture:
    """Two-video fixture with overlap/no-overlap/boundary ASR intervals."""

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
            ObjectDetection(label="person", confidence=0.95, x_min=10, y_min=20, x_max=110, y_max=220, model_source="yolo_world"),
        ),
        "V001_00001_050": (
            ObjectDetection(label="person", confidence=0.91, x_min=15, y_min=25, x_max=115, y_max=225, model_source="yolo_world"),
            ObjectDetection(label="person", confidence=0.35, x_min=120, y_min=30, x_max=200, y_max=230, model_source="co_detr"),
            ObjectDetection(label="car", confidence=0.88, x_min=210, y_min=120, x_max=410, y_max=300, model_source="co_detr"),
        ),
        "V002_00000_025": (
            ObjectDetection(label="bicycle", confidence=0.82, x_min=5, y_min=40, x_max=90, y_max=180),
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
    ocr_hits = (
        FrameSearchHit(frame_id="V001_00000_050", video_id="V001", shot_id=0, raw_score=7.2),
        FrameSearchHit(frame_id="V002_00001_075", video_id="V002", shot_id=1, raw_score=5.1),
    )
    asr_hits = (
        ASRSearchHit(video_id="V001", interval_id="overlap", start_time_sec=1.0, end_time_sec=6.0, raw_score=0.9, text="overlap case"),
        ASRSearchHit(video_id="V001", interval_id="boundary", start_time_sec=10.0, end_time_sec=12.0, raw_score=0.8, text="boundary case"),
        ASRSearchHit(video_id="V002", interval_id="no_overlap", start_time_sec=8.0, end_time_sec=9.0, raw_score=0.7, text="no overlap case"),
    )
    summary_hits = (
        VideoSearchHit(video_id="V001", raw_score=0.85, summary="First video"),
        VideoSearchHit(video_id="V002", raw_score=0.75, summary="Second video"),
    )
    return IntegrationFixture(
        frames=frames,
        objects=objects,
        visual_hits=visual_hits,
        ocr_hits=ocr_hits,
        asr_hits=asr_hits,
        summary_hits=summary_hits,
        missing_metadata_hit=FrameSearchHit(
            frame_id="V999_99999_999", video_id="V999", shot_id=99999, raw_score=0.1
        ),
    )
