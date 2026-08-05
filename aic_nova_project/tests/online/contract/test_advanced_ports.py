from __future__ import annotations

import unittest
from collections.abc import Iterable, Mapping, Sequence

from pydantic import ValidationError

from online.domain.vqa import (
    ASREvidence,
    ImageEvidence,
    OCREvidence,
    SummaryEvidence,
    VLMRequest,
    VLMResponse,
)
from online.ports import (
    EvidenceHydrationPort,
    ImageResolverPort,
    OrderedVisualFrame,
    VLMPort,
    VisualCorpusPort,
    validate_ordered_visual_stream,
)


class MinimalVisualCorpus:
    def __init__(self) -> None:
        self.frame = OrderedVisualFrame(
            frame_id="L21_V001_00000_015",
            video_id="L21_V001",
            shot_id=0,
            local_index=0,
            timestamp_sec=0.0,
            source_frame_idx=0,
            image_rel_path="keyframes/L21_V001/0a.webp",
            vector=(1.0, 0.0),
        )

    def list_video_ids(self) -> Sequence[str]:
        return ("L21_V001",)

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ) -> Iterable[Sequence[OrderedVisualFrame]]:
        return ((self.frame,),)


class MinimalEvidenceHydrator:
    def get_ocr_evidence(self, frame_ids: Sequence[str]) -> Sequence[OCREvidence]:
        return ()

    def get_asr_evidence(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
    ) -> Sequence[ASREvidence]:
        return ()

    def get_summary_evidence(
        self,
        video_ids: Sequence[str],
    ) -> Sequence[SummaryEvidence]:
        return ()


class MinimalImageResolver:
    def resolve_images(self, frame_ids: Sequence[str]) -> Mapping[str, ImageEvidence]:
        return {}


class MinimalVLM:
    def answer(self, request: VLMRequest) -> VLMResponse:
        return VLMResponse(
            status="insufficient_evidence",
            answer=None,
            answer_type="short_text",
            confidence="low",
            evidence_ids=(),
        )


class AdvancedPortContractTests(unittest.TestCase):
    def test_minimal_doubles_conform_to_runtime_protocols(self) -> None:
        self.assertIsInstance(MinimalVisualCorpus(), VisualCorpusPort)
        self.assertIsInstance(MinimalEvidenceHydrator(), EvidenceHydrationPort)
        self.assertIsInstance(MinimalImageResolver(), ImageResolverPort)
        self.assertIsInstance(MinimalVLM(), VLMPort)

    def test_visual_record_is_sdk_neutral_canonical_finite_and_normalized(self) -> None:
        record = MinimalVisualCorpus().frame
        self.assertEqual(record.local_index, 0)
        self.assertEqual(record.vector, (1.0, 0.0))
        self.assertNotIn("pk", OrderedVisualFrame.model_fields)
        self.assertNotIn("client", OrderedVisualFrame.model_fields)
        self.assertEqual(
            OrderedVisualFrame.model_validate_json(record.model_dump_json()),
            record,
        )

        base = record.model_dump()
        for field, value in (
            ("frame_id", "shot_00000_pos_015"),
            ("video_id", "L21_V002"),
            ("shot_id", 2),
            ("source_frame_idx", True),
            ("vector", (2.0, 0.0)),
            ("vector", (float("nan"), 0.0)),
            ("local_index", True),
        ):
            with self.assertRaises(ValidationError):
                OrderedVisualFrame.model_validate({**base, field: value})

    def test_ordered_batches_reconstruct_local_order(self) -> None:
        corpus = MinimalVisualCorpus()
        batches = tuple(corpus.iter_ordered_frame_embedding_batches("L21_V001", 1))
        flattened = tuple(frame for batch in batches for frame in batch)
        self.assertEqual(tuple(frame.local_index for frame in flattened), (0,))
        self.assertEqual(tuple(frame.video_id for frame in flattened), ("L21_V001",))

    def test_stream_validator_accepts_ordered_batches_and_preserves_order(self) -> None:
        frames = (
            OrderedVisualFrame(
                frame_id="L21_V001_00000_015",
                video_id="L21_V001",
                shot_id=0,
                local_index=0,
                timestamp_sec=0.0,
                source_frame_idx=0,
                image_rel_path="keyframes/L21_V001/0a.webp",
                vector=(1.0, 0.0),
            ),
            OrderedVisualFrame(
                frame_id="L21_V001_00000_050",
                video_id="L21_V001",
                shot_id=0,
                local_index=1,
                timestamp_sec=1.0 / 30.0,
                source_frame_idx=0,
                image_rel_path="keyframes/L21_V001/0b.webp",
                vector=(0.0, 1.0),
            ),
            OrderedVisualFrame(
                frame_id="L21_V001_00001_050",
                video_id="L21_V001",
                shot_id=1,
                local_index=2,
                timestamp_sec=2.0,
                source_frame_idx=60,
                image_rel_path="keyframes/L21_V001/1.webp",
                vector=(1.0, 0.0),
            ),
        )
        result = validate_ordered_visual_stream("L21_V001", ((frames[0],), frames[1:]))
        self.assertEqual(result, frames)
        self.assertEqual(result[0].source_frame_idx, result[1].source_frame_idx)

    def test_stream_validator_rejects_wrong_video_duplicates_gaps_reordered_batches_and_dimension_changes(
        self,
    ) -> None:
        frame0 = MinimalVisualCorpus().frame
        frame1 = OrderedVisualFrame(
            frame_id="L21_V001_00000_050",
            video_id="L21_V001",
            shot_id=0,
            local_index=1,
            timestamp_sec=1.0 / 30.0,
            source_frame_idx=0,
            image_rel_path="keyframes/L21_V001/0b.webp",
            vector=(0.0, 1.0),
        )
        frame2 = OrderedVisualFrame(
            frame_id="L21_V001_00001_050",
            video_id="L21_V001",
            shot_id=1,
            local_index=2,
            timestamp_sec=2.0,
            source_frame_idx=60,
            image_rel_path="keyframes/L21_V001/1.webp",
            vector=(1.0, 0.0),
        )
        wrong_video = frame1.model_copy(
            update={"video_id": "L21_V002", "frame_id": "L21_V002_00000_050"}
        )
        duplicate_frame_id = frame0.model_copy(update={"local_index": 1})
        duplicate_local_index = frame1.model_copy(
            update={"frame_id": "L21_V001_00000_015", "shot_id": 0, "local_index": 0}
        )
        dimension_change = OrderedVisualFrame(
            frame_id="L21_V001_00000_050",
            video_id="L21_V001",
            shot_id=0,
            local_index=1,
            timestamp_sec=1.0,
            source_frame_idx=30,
            image_rel_path="keyframes/L21_V001/dim.webp",
            vector=(1.0, 0.0, 0.0),
        )

        invalid_streams = (
            ((wrong_video,),),
            ((frame0,), (duplicate_frame_id,)),
            ((frame0,), (duplicate_local_index,)),
            ((frame0,), (frame2,)),
            ((frame1,), (frame0,)),
            ((frame0,), (dimension_change,)),
        )
        for batches in invalid_streams:
            with self.assertRaises(ValueError):
                validate_ordered_visual_stream("L21_V001", batches)

        with self.assertRaises(ValueError):
            validate_ordered_visual_stream("L21_V001", ((frame1, frame0),))


if __name__ == "__main__":
    unittest.main()
