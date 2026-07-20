from __future__ import annotations

from online.domain.candidates import CandidateDiagnostics, FusedFrameCandidate
import pytest

from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DimensionMismatchError,
    InvalidQueryError,
    MissingMetadataError,
    ResourceUnavailableError,
)
from online.domain.vqa import ASREvidence, ImageEvidence, OCREvidence, SummaryEvidence, VQAEvidenceBudget, VQAAnswerType, VQAQuestion
from online.ports.records import FrameMetadata
from online.vqa.evidence_selector import EvidenceSelector, map_evidence_budget


def candidate(frame_id: str, video_id: str, score: float, timestamp: float) -> FusedFrameCandidate:
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=0,
        timestamp_sec=timestamp,
        final_score=score,
        branch_scores={},
        evidence=(),
        diagnostics=CandidateDiagnostics(),
    )


class Metadata:
    def __init__(self) -> None:
        self.frames = {
            "V001": tuple(FrameMetadata(frame_id=f"V001_00000_{index:03d}", video_id="V001", shot_id=0, timestamp_sec=float(index)) for index in range(5)),
            "V002": tuple(FrameMetadata(frame_id=f"V002_00000_{index:03d}", video_id="V002", shot_id=0, timestamp_sec=float(index)) for index in range(3)),
        }

    def get_frames_by_ids(self, frame_ids):
        return {item.frame_id: item for values in self.frames.values() for item in values if item.frame_id in frame_ids}

    def get_ordered_frames_by_video(self, video_id):
        return self.frames[video_id]


class Images:
    def __init__(self, *, missing: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.calls: list[tuple[str, ...]] = []

    def resolve_images(self, frame_ids):
        self.calls.append(tuple(frame_ids))
        return {
            frame_id: ImageEvidence(
                evidence_id=f"image-{frame_id}",
                video_id=frame_id.split("_")[0],
                frame_id=frame_id,
                shot_id=0,
                timestamp_sec=float(frame_id[-3:]),
                image_reference=f"fixture://images/{frame_id}",
            )
            for frame_id in frame_ids
            if frame_id not in self.missing
        }


class Hydrator:
    def __init__(self, *, fail_ocr: bool = False) -> None:
        self.fail_ocr = fail_ocr
        self.ocr_calls: list[tuple[str, ...]] = []
        self.asr_calls: list[tuple[str, float, float]] = []
        self.summary_calls: list[tuple[str, ...]] = []

    def get_ocr_evidence(self, frame_ids):
        self.ocr_calls.append(tuple(frame_ids))
        if self.fail_ocr:
            raise ResourceUnavailableError("ocr unavailable")
        return tuple(OCREvidence(evidence_id=f"ocr-{frame_id}", video_id=frame_id.split("_")[0], frame_id=frame_id, text="OCR") for frame_id in frame_ids)

    def get_asr_evidence(self, video_id, start_sec, end_sec):
        self.asr_calls.append((video_id, start_sec, end_sec))
        return (
            ASREvidence(evidence_id=f"asr-{video_id}-{start_sec}", video_id=video_id, interval_id=f"i-{start_sec}", start_time_sec=start_sec, end_time_sec=end_sec, text="ASR"),
        )

    def get_summary_evidence(self, video_ids):
        self.summary_calls.append(tuple(video_ids))
        return tuple(SummaryEvidence(evidence_id=f"summary-{video_id}", video_id=video_id, text="summary") for video_id in video_ids)


QUESTION = VQAQuestion(question_id="q1", question="Ai xuất hiện?", answer_type=VQAAnswerType.SHORT_TEXT)


def test_public_budget_maps_every_default_and_override() -> None:
    budget = VQAEvidenceBudget(max_videos=2, max_primary_frames_per_video=2, max_primary_frames_total=4, max_images_total=7, max_ocr_chars=10, max_asr_chars=11, max_summary_chars_per_video=3, max_summary_chars_total=6, max_text_chars_total=20, asr_window_sec=2.5)
    policy = map_evidence_budget(budget)
    assert (policy.max_videos, policy.max_primary_per_video, policy.max_primary_total, policy.max_images_total) == (2, 2, 4, 7)
    assert (policy.ocr_chars, policy.asr_chars, policy.summary_chars_per_video, policy.summary_chars_total, policy.text_chars_total, policy.asr_window_seconds) == (10, 11, 3, 6, 20, 2.5)


def test_selector_hydrates_only_selected_evidence_and_is_deterministic() -> None:
    frames = (
        candidate("V001_00000_002", "V001", 1.0, 2.0),
        candidate("V002_00000_001", "V002", 0.9, 1.0),
        candidate("V001_00000_004", "V001", 0.8, 4.0),
    )
    images = Images()
    hydrator = Hydrator()
    selector = EvidenceSelector(metadata_reader=Metadata(), image_resolver=images, evidence_hydrator=hydrator)
    budget = VQAEvidenceBudget(max_videos=2, max_primary_frames_per_video=1, max_primary_frames_total=2, max_images_total=4)

    first = selector.select(QUESTION, frames, budget)
    second = selector.select(QUESTION, tuple(reversed(frames)), budget)

    assert first == second
    assert first.selected_primary_count == 2
    assert first.selected_image_count == 4
    assert len(images.calls[0]) == 4
    assert set(hydrator.ocr_calls[0]) == set(images.calls[0])
    assert {call[0] for call in hydrator.asr_calls[:2]} == {"V001", "V002"}
    assert hydrator.summary_calls[0] == ("V001", "V002")
    assert len({item.evidence_id for item in first.evidence}) == len(first.evidence)


def test_missing_image_and_optional_text_failure_are_reported_without_fabrication() -> None:
    frame = candidate("V001_00000_002", "V001", 1.0, 2.0)
    images = Images(missing={frame.frame_id})
    selector = EvidenceSelector(metadata_reader=Metadata(), image_resolver=images, evidence_hydrator=Hydrator(fail_ocr=True))

    budget = VQAEvidenceBudget(
        max_videos=1,
        max_primary_frames_per_video=1,
        max_primary_frames_total=1,
        max_images_total=1,
    )
    result = selector.select(QUESTION, (frame,), budget)

    assert result.selected_image_count == 0
    assert result.missing_count == 1
    assert "MISSING_IMAGE_EVIDENCE" in result.warnings
    assert "OCR_RESOURCE_UNAVAILABLE" in result.warnings
    assert all(not isinstance(item, OCREvidence) for item in result.evidence)


SMALL_BUDGET = VQAEvidenceBudget(
    max_videos=1,
    max_primary_frames_per_video=1,
    max_primary_frames_total=1,
    max_images_total=1,
)


class ConfigurableHydrator(Hydrator):
    def __init__(self, *, ocr=(), asr=(), summary=()) -> None:
        super().__init__()
        self.outputs = {"ocr": ocr, "asr": asr, "summary": summary}

    @staticmethod
    def _result(value):
        if isinstance(value, Exception):
            raise value
        return value

    def get_ocr_evidence(self, frame_ids):
        return self._result(self.outputs["ocr"])

    def get_asr_evidence(self, video_id, start_sec, end_sec):
        return self._result(self.outputs["asr"])

    def get_summary_evidence(self, video_ids):
        return self._result(self.outputs["summary"])


def select_with(*, hydrator=None, metadata=None, images=None, budget=SMALL_BUDGET):
    service = EvidenceSelector(
        metadata_reader=metadata or Metadata(),
        image_resolver=images or Images(),
        evidence_hydrator=hydrator or ConfigurableHydrator(),
    )
    return service.select(
        QUESTION,
        (candidate("V001_00000_002", "V001", 1.0, 2.0),),
        budget,
    )


@pytest.mark.parametrize(
    "asr",
    (
        (object(),),
        (
            ASREvidence(
                evidence_id="asr-wrong-video",
                video_id="V002",
                interval_id="i1",
                start_time_sec=0,
                end_time_sec=1,
                text="x",
            ),
        ),
        (
            ASREvidence(
                evidence_id="asr-outside",
                video_id="V001",
                interval_id="i2",
                start_time_sec=10,
                end_time_sec=11,
                text="x",
            ),
        ),
    ),
)
def test_asr_invalid_item_video_or_window_is_contract_mismatch(asr) -> None:
    with pytest.raises(ContractMismatchError):
        select_with(hydrator=ConfigurableHydrator(asr=asr))


def test_summary_for_unrequested_video_is_contract_mismatch() -> None:
    summary = SummaryEvidence(evidence_id="summary-v2", video_id="V002", text="x")
    with pytest.raises(ContractMismatchError):
        select_with(hydrator=ConfigurableHydrator(summary=(summary,)))


@pytest.mark.parametrize("field", ("video_id", "shot_id", "timestamp_sec"))
def test_image_metadata_must_match_requested_frame(field: str) -> None:
    valid = ImageEvidence(
        evidence_id="image-1",
        video_id="V001",
        frame_id="V001_00000_002",
        shot_id=0,
        timestamp_sec=2,
        image_reference="fixture://image/1",
    )
    wrong_value = {"video_id": "V002", "shot_id": 1, "timestamp_sec": 3.0}[field]
    invalid = valid.model_copy(update={field: wrong_value})

    class InvalidImages:
        def resolve_images(self, frame_ids):
            return {"V001_00000_002": invalid}

    with pytest.raises(ContractMismatchError):
        select_with(images=InvalidImages())


@pytest.mark.parametrize(
    "frames",
    (
        (object(),),
        (FrameMetadata(frame_id="V002_00000_001", video_id="V002", shot_id=0, timestamp_sec=1),),
        (
            FrameMetadata(frame_id="V001_00000_002", video_id="V001", shot_id=0, timestamp_sec=2),
            FrameMetadata(frame_id="V001_00000_002", video_id="V001", shot_id=0, timestamp_sec=2),
        ),
    ),
)
def test_invalid_metadata_stream_is_contract_mismatch(frames) -> None:
    class InvalidMetadata(Metadata):
        def get_ordered_frames_by_video(self, video_id):
            return frames

    with pytest.raises(ContractMismatchError):
        select_with(metadata=InvalidMetadata())


@pytest.mark.parametrize("stage", ("ocr", "asr", "summary"))
@pytest.mark.parametrize(
    ("error", "code"),
    (
        (ResourceUnavailableError("unavailable"), "RESOURCE_UNAVAILABLE"),
        (BranchTimeoutError("timeout"), "BRANCH_TIMEOUT"),
    ),
)
def test_optional_backend_outage_degrades_without_fabrication(stage, error, code) -> None:
    values = {"ocr": (), "asr": (), "summary": ()}
    values[stage] = error
    result = select_with(hydrator=ConfigurableHydrator(**values))
    assert f"{stage.upper()}_{code}" in result.warnings
    assert result.selected_text_count == 0


@pytest.mark.parametrize(
    "error",
    (
        InvalidQueryError("invalid"),
        ContractMismatchError("contract"),
        DimensionMismatchError("dimension"),
        MissingMetadataError("metadata"),
    ),
)
def test_contract_family_errors_are_not_degraded(error) -> None:
    with pytest.raises(type(error)):
        select_with(hydrator=ConfigurableHydrator(ocr=error))


def test_truncated_text_counts_one_dropped_evidence_record() -> None:
    ocr = OCREvidence(
        evidence_id="ocr-1",
        video_id="V001",
        frame_id="V001_00000_002",
        text="abcdef",
    )
    budget = VQAEvidenceBudget(
        max_videos=1,
        max_primary_frames_per_video=1,
        max_primary_frames_total=1,
        max_images_total=1,
        max_ocr_chars=2,
        max_asr_chars=2,
        max_summary_chars_per_video=1,
        max_summary_chars_total=1,
        max_text_chars_total=2,
    )
    result = select_with(hydrator=ConfigurableHydrator(ocr=(ocr,)), budget=budget)
    selected = next(item for item in result.evidence if isinstance(item, OCREvidence))
    assert selected.text == "ab"
    assert result.dropped_count == 1
