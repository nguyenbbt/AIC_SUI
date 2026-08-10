"""Production OCR/ASR/summary evidence hydration from Elasticsearch."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from online.config import ElasticsearchResourceConfig
from online.domain.errors import ContractMismatchError, InvalidQueryError
from online.domain.vqa import ASREvidence, OCREvidence, SummaryEvidence


class ElasticsearchEvidenceBackend(Protocol):
    def find_documents(
        self,
        index: str,
        filters: Mapping[str, object],
        source_fields: Sequence[str],
        *,
        limit: int = 2,
    ) -> Sequence[Mapping[str, object]]: ...

    def find_documents_overlapping_interval(
        self,
        *,
        index: str,
        video_id: str,
        start_sec: float,
        end_sec: float,
        source_fields: Sequence[str],
        limit: int,
    ) -> Sequence[Mapping[str, object]]: ...


class ElasticsearchEvidenceHydrator:
    """Hydrate stable VQA evidence without issuing retrieval/ranking queries."""

    def __init__(
        self,
        config: ElasticsearchResourceConfig,
        *,
        backend: ElasticsearchEvidenceBackend,
    ) -> None:
        if not callable(getattr(backend, "find_documents", None)) or not callable(
            getattr(backend, "find_documents_overlapping_interval", None)
        ):
            raise TypeError("backend must implement ElasticsearchEvidenceBackend")
        self.config = config
        self._backend = backend

    def get_ocr_evidence(self, frame_ids: Sequence[str]) -> Sequence[OCREvidence]:
        ids = _validate_ids(frame_ids, "frame_ids")
        output: list[OCREvidence] = []
        for frame_id in ids:
            records = self._backend.find_documents(
                self.config.ocr_index,
                {"frame_id": frame_id},
                ("frame_id", "video_id", "ocr_text_concat"),
                limit=2,
            )
            if len(records) > 1:
                raise ContractMismatchError("OCR frame_id is not unique in Elasticsearch")
            if records:
                record = records[0]
                actual_frame_id = _text(record, "frame_id")
                if actual_frame_id != frame_id:
                    raise ContractMismatchError("OCR lookup returned a different frame_id")
                output.append(
                    OCREvidence(
                        evidence_id=f"ocr:{frame_id}",
                        video_id=_text(record, "video_id"),
                        frame_id=actual_frame_id,
                        text=_text(record, "ocr_text_concat"),
                    )
                )
        return tuple(output)

    def get_asr_evidence(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
    ) -> Sequence[ASREvidence]:
        _validate_video_and_window(video_id, start_sec, end_sec)
        records = self._backend.find_documents_overlapping_interval(
            index=self.config.asr_index,
            video_id=video_id,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            source_fields=(
                "video_id",
                "interval_id",
                "start_time_sec",
                "end_time_sec",
                "cleaned_text",
            ),
            limit=self.config.evidence_lookup_limit,
        )
        output_items: list[ASREvidence] = []
        for record in records:
            actual_video_id = _text(record, "video_id")
            if actual_video_id != video_id:
                raise ContractMismatchError("ASR evidence lookup returned a different video_id")
            interval_id = _text(record, "interval_id")
            output_items.append(
                ASREvidence(
                    evidence_id=f"asr:{video_id}:{interval_id}",
                    video_id=actual_video_id,
                    interval_id=interval_id,
                    start_time_sec=_number(record, "start_time_sec"),
                    end_time_sec=_number(record, "end_time_sec"),
                    text=_text(record, "cleaned_text"),
                )
            )
        output = tuple(output_items)
        if any(item.video_id != video_id for item in output):
            raise ContractMismatchError("ASR evidence lookup returned a different video_id")
        identities = tuple((item.video_id, item.interval_id) for item in output)
        if len(identities) != len(set(identities)):
            raise ContractMismatchError("ASR interval identity is duplicated")
        if any(item.end_time_sec < start_sec or item.start_time_sec > end_sec for item in output):
            raise ContractMismatchError("ASR backend returned a non-overlapping interval")
        return output

    def get_summary_evidence(
        self,
        video_ids: Sequence[str],
    ) -> Sequence[SummaryEvidence]:
        ids = _validate_ids(video_ids, "video_ids")
        output: list[SummaryEvidence] = []
        for video_id in ids:
            records = self._backend.find_documents(
                self.config.summary_index,
                {"video_id": video_id},
                ("video_id", "summary"),
                limit=2,
            )
            if len(records) > 1:
                raise ContractMismatchError("Summary video_id is not unique in Elasticsearch")
            if records:
                record = records[0]
                actual_video_id = _text(record, "video_id")
                if actual_video_id != video_id:
                    raise ContractMismatchError("Summary lookup returned a different video_id")
                output.append(
                    SummaryEvidence(
                        evidence_id=f"summary:{video_id}",
                        video_id=video_id,
                        text=_text(record, "summary"),
                    )
                )
        return tuple(output)


def _validate_ids(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise InvalidQueryError(f"{field} must be a sequence of strings")
    try:
        result = tuple(dict.fromkeys(values))
    except (TypeError, ValueError) as exc:
        raise InvalidQueryError(f"{field} must be a sequence of strings") from exc
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in result
    ):
        raise InvalidQueryError(f"{field} must contain canonical non-empty strings")
    return result


def _validate_video_and_window(video_id: str, start_sec: float, end_sec: float) -> None:
    if not isinstance(video_id, str) or not video_id.strip() or video_id != video_id.strip():
        raise InvalidQueryError("video_id must be a canonical string")
    if (
        isinstance(start_sec, bool)
        or isinstance(end_sec, bool)
        or not isinstance(start_sec, (int, float))
        or not isinstance(end_sec, (int, float))
        or not math.isfinite(float(start_sec))
        or not math.isfinite(float(end_sec))
        or float(start_sec) < 0.0
        or float(end_sec) < float(start_sec)
    ):
        raise InvalidQueryError("ASR evidence window must be finite and ordered")


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractMismatchError(
            "Elasticsearch evidence field must be canonical non-empty text",
            details={"field": field},
        )
    return value


def _number(record: Mapping[str, object], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractMismatchError(
            "Elasticsearch evidence field must be numeric",
            details={"field": field},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ContractMismatchError(
            "Elasticsearch evidence time must be finite and non-negative",
            details={"field": field},
        )
    return number


__all__ = ["ElasticsearchEvidenceBackend", "ElasticsearchEvidenceHydrator"]
