"""Adapter from ranked frames and hydration ports to public VQA evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from online.domain.candidates import FusedFrameCandidate
from online.domain.errors import ContractMismatchError, DataInfrastructureError
from online.domain.vqa import (
    ASREvidence,
    EvidenceType,
    ImageEvidence,
    OCREvidence,
    SummaryEvidence,
    VQAEvidence,
    VQAEvidenceBudget,
    VQAQuestion,
)
from online.ports.evidence import EvidenceHydrationPort
from online.ports.images import ImageResolverPort
from online.ports.metadata import MetadataReaderPort

from .budget import EvidenceBudgetPolicy
from .selection import (
    _TextEvidenceChunk,
    apply_text_budget,
    filter_asr_chunks_for_windows,
    select_neighbor_frames,
    select_primary_frames,
)


@dataclass(frozen=True, slots=True)
class EvidenceSelectionResult:
    evidence: tuple[VQAEvidence, ...]
    retrieved_frame_count: int
    selected_primary_count: int
    selected_image_count: int
    selected_text_count: int
    dropped_count: int
    missing_count: int
    warnings: tuple[str, ...]


def map_evidence_budget(budget: VQAEvidenceBudget) -> EvidenceBudgetPolicy:
    """Explicitly map every public DD-030 budget field to the pure selector."""

    if not isinstance(budget, VQAEvidenceBudget):
        raise ContractMismatchError("budget must be a validated VQAEvidenceBudget")
    return EvidenceBudgetPolicy(
        max_videos=budget.max_videos,
        max_primary_per_video=budget.max_primary_frames_per_video,
        max_primary_total=budget.max_primary_frames_total,
        max_images_total=budget.max_images_total,
        ocr_chars=budget.max_ocr_chars,
        asr_chars=budget.max_asr_chars,
        summary_chars_per_video=budget.max_summary_chars_per_video,
        summary_chars_total=budget.max_summary_chars_total,
        text_chars_total=budget.max_text_chars_total,
        asr_window_seconds=budget.asr_window_sec,
    )


class EvidenceSelector:
    def __init__(
        self,
        *,
        metadata_reader: MetadataReaderPort,
        image_resolver: ImageResolverPort,
        evidence_hydrator: EvidenceHydrationPort,
    ) -> None:
        self._metadata_reader = metadata_reader
        self._image_resolver = image_resolver
        self._evidence_hydrator = evidence_hydrator

    def select(
        self,
        question: VQAQuestion,
        ranked_candidates: Sequence[FusedFrameCandidate],
        budget: VQAEvidenceBudget,
    ) -> EvidenceSelectionResult:
        if not isinstance(question, VQAQuestion):
            raise ContractMismatchError("question must be a validated VQAQuestion")
        candidates = tuple(ranked_candidates)
        if any(not isinstance(item, FusedFrameCandidate) for item in candidates):
            raise ContractMismatchError("ranked_candidates contains an invalid value")
        policy = map_evidence_budget(budget)
        primary = select_primary_frames(candidates, policy)
        video_ids = tuple(dict.fromkeys(item.video_id for item in primary))

        ordered_frames = {
            video_id: tuple(self._metadata_reader.get_ordered_frames_by_video(video_id))
            for video_id in video_ids
        }
        neighbors = select_neighbor_frames(primary, ordered_frames, policy)
        requested_frames = tuple(primary) + neighbors
        requested_ids = tuple(item.frame_id for item in requested_frames)
        images_by_frame = self._image_resolver.resolve_images(requested_ids)
        self._validate_image_mapping(images_by_frame, requested_ids)
        images = tuple(
            images_by_frame[frame_id]
            for frame_id in requested_ids
            if frame_id in images_by_frame
        )

        warnings: list[str] = []
        hydrated_text: list[OCREvidence | ASREvidence | SummaryEvidence] = []
        attempted_text_count = 0

        try:
            ocr = tuple(self._evidence_hydrator.get_ocr_evidence(tuple(item.frame_id for item in images)))
            self._validate_ocr(ocr, {item.frame_id for item in images})
            hydrated_text.extend(ocr)
            attempted_text_count += len(ocr)
        except DataInfrastructureError as exc:
            if isinstance(exc, ContractMismatchError):
                raise
            warnings.append(f"OCR_{exc.code.value}")

        asr: tuple[ASREvidence, ...] = ()
        try:
            asr = self._hydrate_asr(primary, policy)
            hydrated_text.extend(asr)
            attempted_text_count += len(asr)
        except DataInfrastructureError as exc:
            if isinstance(exc, ContractMismatchError):
                raise
            warnings.append(f"ASR_{exc.code.value}")

        try:
            summaries = tuple(self._evidence_hydrator.get_summary_evidence(video_ids))
            self._validate_summaries(summaries, set(video_ids))
            hydrated_text.extend(summaries)
            attempted_text_count += len(summaries)
        except DataInfrastructureError as exc:
            if isinstance(exc, ContractMismatchError):
                raise
            warnings.append(f"SUMMARY_{exc.code.value}")

        chunks, records_by_id = self._to_chunks(hydrated_text)
        budgeted = apply_text_budget(chunks, video_ids, policy)
        selected_text = tuple(
            self._replace_text(records_by_id[item.stable_id], item.text)
            for item in budgeted
        )
        evidence, collision_count = self._deduplicate((*images, *selected_text))
        missing_count = len(requested_ids) - len(images)
        dropped_count = max(0, attempted_text_count - len(selected_text)) + collision_count
        if missing_count:
            warnings.append("MISSING_IMAGE_EVIDENCE")
        return EvidenceSelectionResult(
            evidence=evidence,
            retrieved_frame_count=len(candidates),
            selected_primary_count=len(primary),
            selected_image_count=sum(item.evidence_type is EvidenceType.IMAGE for item in evidence),
            selected_text_count=sum(item.evidence_type is not EvidenceType.IMAGE for item in evidence),
            dropped_count=dropped_count,
            missing_count=missing_count,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _hydrate_asr(
        self,
        primary: Sequence[FusedFrameCandidate],
        policy: EvidenceBudgetPolicy,
    ) -> tuple[ASREvidence, ...]:
        records: dict[str, ASREvidence] = {}
        for frame in primary:
            start = max(0.0, frame.timestamp_sec - policy.asr_window_seconds)
            end = frame.timestamp_sec + policy.asr_window_seconds
            for item in self._evidence_hydrator.get_asr_evidence(frame.video_id, start, end):
                if item.video_id != frame.video_id:
                    raise ContractMismatchError("ASR hydrator returned evidence for an unrequested video")
                if item.end_time_sec < start or item.start_time_sec > end:
                    raise ContractMismatchError("ASR hydrator returned evidence outside the requested window")
                records.setdefault(item.evidence_id, item)
        chunks, by_id = self._to_chunks(tuple(records.values()))
        filtered = filter_asr_chunks_for_windows(chunks, primary, policy)
        return tuple(by_id[item.stable_id] for item in filtered)  # type: ignore[return-value]

    @staticmethod
    def _validate_image_mapping(images: object, requested_ids: Sequence[str]) -> None:
        if not hasattr(images, "items"):
            raise ContractMismatchError("image resolver returned a non-mapping value")
        allowed = set(requested_ids)
        for key, item in images.items():  # type: ignore[union-attr]
            if key not in allowed or not isinstance(item, ImageEvidence) or item.frame_id != key:
                raise ContractMismatchError("image resolver returned an invalid or unrequested image")

    @staticmethod
    def _validate_ocr(records: Sequence[OCREvidence], frame_ids: set[str]) -> None:
        if any(not isinstance(item, OCREvidence) or item.frame_id not in frame_ids for item in records):
            raise ContractMismatchError("OCR hydrator returned invalid or unrequested evidence")

    @staticmethod
    def _validate_summaries(records: Sequence[SummaryEvidence], video_ids: set[str]) -> None:
        if any(not isinstance(item, SummaryEvidence) or item.video_id not in video_ids for item in records):
            raise ContractMismatchError("summary hydrator returned invalid or unrequested evidence")

    @staticmethod
    def _to_chunks(
        records: Sequence[OCREvidence | ASREvidence | SummaryEvidence],
    ) -> tuple[tuple[_TextEvidenceChunk, ...], dict[str, OCREvidence | ASREvidence | SummaryEvidence]]:
        chunks: list[_TextEvidenceChunk] = []
        by_id: dict[str, OCREvidence | ASREvidence | SummaryEvidence] = {}
        type_rank = {EvidenceType.OCR: 0, EvidenceType.ASR: 1, EvidenceType.SUMMARY: 2}
        ordered = sorted(records, key=lambda item: (type_rank[item.evidence_type], item.evidence_id))
        for source_order, item in enumerate(ordered):
            if item.evidence_id in by_id:
                continue
            by_id[item.evidence_id] = item
            chunks.append(
                _TextEvidenceChunk(
                    stable_id=item.evidence_id,
                    evidence_type=item.evidence_type.value,  # type: ignore[arg-type]
                    source_rank=type_rank[item.evidence_type],
                    source_order=source_order,
                    text=item.text,
                    video_id=item.video_id,
                    start_time_sec=item.start_time_sec if isinstance(item, ASREvidence) else None,
                    end_time_sec=item.end_time_sec if isinstance(item, ASREvidence) else None,
                )
            )
        return tuple(chunks), by_id

    @staticmethod
    def _replace_text(
        record: OCREvidence | ASREvidence | SummaryEvidence,
        text: str,
    ) -> OCREvidence | ASREvidence | SummaryEvidence:
        return record if record.text == text else record.model_copy(update={"text": text})

    @staticmethod
    def _deduplicate(evidence: Sequence[VQAEvidence]) -> tuple[tuple[VQAEvidence, ...], int]:
        output: list[VQAEvidence] = []
        seen: set[str] = set()
        for item in evidence:
            if item.evidence_id not in seen:
                output.append(item)
                seen.add(item.evidence_id)
        return tuple(output), len(evidence) - len(output)


__all__ = ["EvidenceSelectionResult", "EvidenceSelector", "map_evidence_budget"]
