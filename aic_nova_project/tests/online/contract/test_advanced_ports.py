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
)


class MinimalVisualCorpus:
    def __init__(self) -> None:
        self.frame = OrderedVisualFrame(
            frame_id="V001_00000_015",
            video_id="V001",
            shot_id=0,
            local_index=0,
            timestamp_sec=1.5,
            vector=(1.0, 0.0),
        )

    def list_video_ids(self) -> Sequence[str]:
        return ("V001",)

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
            ("video_id", "V002"),
            ("vector", (2.0, 0.0)),
            ("vector", (float("nan"), 0.0)),
            ("local_index", True),
        ):
            with self.assertRaises(ValidationError):
                OrderedVisualFrame.model_validate({**base, field: value})

    def test_ordered_batches_reconstruct_local_order(self) -> None:
        corpus = MinimalVisualCorpus()
        batches = tuple(corpus.iter_ordered_frame_embedding_batches("V001", 1))
        flattened = tuple(frame for batch in batches for frame in batch)
        self.assertEqual(tuple(frame.local_index for frame in flattened), (0,))
        self.assertEqual(tuple(frame.video_id for frame in flattened), ("V001",))


if __name__ == "__main__":
    unittest.main()
