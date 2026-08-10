"""Read-only OCR, ASR and summary evidence hydration boundary."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from online.domain.vqa import ASREvidence, OCREvidence, SummaryEvidence


@runtime_checkable
class EvidenceHydrationPort(Protocol):
    def get_ocr_evidence(self, frame_ids: Sequence[str]) -> Sequence[OCREvidence]: ...

    def get_asr_evidence(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
    ) -> Sequence[ASREvidence]: ...

    def get_summary_evidence(
        self,
        video_ids: Sequence[str],
    ) -> Sequence[SummaryEvidence]: ...


EvidenceReaderPort = EvidenceHydrationPort
