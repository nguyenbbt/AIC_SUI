from __future__ import annotations

import math
import unittest

from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.domain.identifiers import validate_canonical_frame_id
from online.domain.vqa import (
    VLMRequest,
    VLMResponse,
    VLMResponseStatus,
    VQAEvidence,
)
from online.ports.encoders import TextEncoderPort
from online.ports.evidence import EvidenceHydrationPort
from online.ports.images import ImageResolverPort
from online.ports.visual_corpus import VisualCorpusPort, validate_ordered_visual_stream
from online.ports.vlm import VLMPort
from online.testing import (
    AdvancedFakeBehavior,
    FakeVLMMode,
    build_advanced_modes_fixture,
)


class AdvancedFixtureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = build_advanced_modes_fixture()

    def _evidence_by_id(self) -> dict[str, VQAEvidence]:
        records = (
            tuple(self.fixture.images_by_frame_id.values())
            + self.fixture.ocr_evidence
            + self.fixture.asr_evidence
            + self.fixture.summary_evidence
        )
        return {record.evidence_id: record for record in records}

    def _vlm_request(
        self,
        evidence_ids: tuple[str, ...] | None = None,
    ) -> VLMRequest:
        requested_ids = (
            self.fixture.expected_vqa_answer_evidence_ids
            if evidence_ids is None
            else evidence_ids
        )
        evidence_by_id = self._evidence_by_id()
        return VLMRequest(
            request_id="vlm-wave2-contract",
            question=self.fixture.vqa_question,
            evidence=tuple(evidence_by_id[evidence_id] for evidence_id in requested_ids),
        )

    def test_all_fakes_conform_to_runtime_protocols(self) -> None:
        self.assertIsInstance(self.fixture.text_encoder(), TextEncoderPort)
        self.assertIsInstance(self.fixture.visual_corpus(), VisualCorpusPort)
        self.assertIsInstance(
            self.fixture.evidence_hydrator(), EvidenceHydrationPort
        )
        self.assertIsInstance(self.fixture.image_resolver(), ImageResolverPort)
        self.assertIsInstance(self.fixture.vlm(), VLMPort)

    def test_fixture_is_deterministic_across_builds(self) -> None:
        self.assertEqual(self.fixture, build_advanced_modes_fixture())

    def test_expected_vqa_id_tuples_are_deterministic_unique_and_related(
        self,
    ) -> None:
        rebuilt = build_advanced_modes_fixture()
        selected = self.fixture.expected_vqa_selected_evidence_ids
        answer = self.fixture.expected_vqa_answer_evidence_ids

        self.assertEqual(
            selected,
            (
                "image:L21_V001_00004_050",
                "image:L21_V002_00002_050",
                "image:L21_V001_00003_050",
                "ocr:L21_V001_00004_050",
                "ocr:L21_V002_00002_050",
                "asr:L21_V001:interval-1",
                "asr:L21_V002:interval-1",
                "summary:L21_V001",
                "summary:L21_V002",
            ),
        )
        self.assertEqual(
            answer,
            (
                "image:L21_V001_00004_050",
                "ocr:L21_V001_00004_050",
                "asr:L21_V001:interval-1",
                "summary:L21_V001",
            ),
        )
        self.assertEqual(selected, rebuilt.expected_vqa_selected_evidence_ids)
        self.assertEqual(answer, rebuilt.expected_vqa_answer_evidence_ids)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(answer), len(set(answer)))
        self.assertTrue(set(answer).issubset(selected))

    def test_expected_vqa_ids_reference_real_fixture_evidence(self) -> None:
        known_ids = set(self._evidence_by_id())
        self.assertTrue(
            set(self.fixture.expected_vqa_selected_evidence_ids).issubset(known_ids)
        )
        self.assertTrue(
            set(self.fixture.expected_vqa_answer_evidence_ids).issubset(known_ids)
        )

    def test_event_encoder_mapping_is_explicit_normalized_and_safe_logged(self) -> None:
        encoder = self.fixture.text_encoder()
        texts = tuple(event.text for event in self.fixture.trake_query.events)
        vectors = encoder.encode_texts(texts)
        self.assertEqual(vectors, tuple(self.fixture.event_vectors[text] for text in texts))
        self.assertTrue(
            all(math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0) for vector in vectors)
        )
        self.assertEqual(encoder.calls[0].text_count, 3)
        self.assertNotIn("vector", repr(encoder.calls).lower())
        with self.assertRaises(ContractMismatchError):
            encoder.encode_texts(("Không có trong fixture",))

    def test_canonical_ids_dimensions_and_vector_norms_are_valid(self) -> None:
        dimensions: set[int] = set()
        for video_id, frames in self.fixture.visual_frames_by_video.items():
            for frame in frames:
                validate_canonical_frame_id(
                    frame.frame_id,
                    video_id=video_id,
                    shot_id=frame.shot_id,
                )
                dimensions.add(len(frame.vector))
                self.assertTrue(
                    math.isclose(
                        math.sqrt(sum(value * value for value in frame.vector)),
                        1.0,
                    )
                )
        self.assertEqual(dimensions, {4})

    def test_visual_batches_reconstruct_complete_local_order(self) -> None:
        corpus = self.fixture.visual_corpus()
        self.assertEqual(
            corpus.list_video_ids(),
            ("L21_V001", "L21_V002", "L21_V003", "L21_V004"),
        )
        for video_id in corpus.list_video_ids():
            frames = validate_ordered_visual_stream(
                video_id,
                corpus.iter_ordered_frame_embedding_batches(video_id, 2),
            )
            self.assertEqual(
                tuple(frame.local_index for frame in frames),
                tuple(range(len(frames))),
            )
        with self.assertRaises(InvalidQueryError):
            corpus.iter_ordered_frame_embedding_batches("L21_V001", True)
        with self.assertRaises(InvalidQueryError):
            corpus.iter_ordered_frame_embedding_batches("unknown", 2)

    def test_trake_edge_cases_are_distinct_and_expected_score_is_hand_computable(self) -> None:
        frames = self.fixture.visual_frames_by_video
        self.assertGreaterEqual(len(frames["L21_V001"]), 6)
        self.assertGreaterEqual(len(frames["L21_V002"]), 6)
        self.assertGreaterEqual(len(frames["L21_V003"]), 6)
        self.assertLess(len(frames["L21_V004"]), len(self.fixture.trake_query.events))
        self.assertEqual(self.fixture.expected_dante_video_id, "L21_V001")
        self.assertEqual(self.fixture.expected_dante_positions, (0, 2, 4))
        expected = 1.0 + 1.0 + 1.0 - 0.001 * ((2 - 0) + (4 - 2))
        self.assertAlmostEqual(self.fixture.expected_dante_score, expected)
        self.assertEqual(
            self.fixture.tied_sequence_positions,
            ((1, 2, 4), (1, 3, 4)),
        )
        # L21_V002 contains the same perfect event vectors, but in reverse order.
        self.assertEqual(frames["L21_V002"][0].vector, self.fixture.event_vectors[
            self.fixture.trake_query.events[2].text
        ])
        self.assertEqual(frames["L21_V002"][4].vector, self.fixture.event_vectors[
            self.fixture.trake_query.events[0].text
        ])

    def test_visual_failures_are_injectable_and_distinct(self) -> None:
        for error_type in (
            ResourceUnavailableError,
            BranchTimeoutError,
            ContractMismatchError,
        ):
            corpus = self.fixture.visual_corpus(
                video_behaviors={
                    "L21_V001": AdvancedFakeBehavior(error_type("simulated"))
                }
            )
            with self.assertRaises(error_type):
                corpus.iter_ordered_frame_embedding_batches("L21_V001", 2)
        unavailable = self.fixture.visual_corpus(
            list_behavior=AdvancedFakeBehavior(
                ResourceUnavailableError("simulated list failure")
            )
        )
        with self.assertRaises(ResourceUnavailableError):
            unavailable.list_video_ids()

    def test_evidence_hydration_never_returns_records_outside_request(self) -> None:
        hydrator = self.fixture.evidence_hydrator()
        requested_frame = self.fixture.ocr_evidence[0].frame_id
        ocr = hydrator.get_ocr_evidence((requested_frame, "L21_V001_00005_050"))
        self.assertEqual(tuple(item.frame_id for item in ocr), (requested_frame,))

        asr = hydrator.get_asr_evidence("L21_V001", 7.0, 8.0)
        self.assertEqual(tuple(item.video_id for item in asr), ("L21_V001",))
        self.assertTrue(
            all(item.end_time_sec >= 7.0 and item.start_time_sec <= 8.0 for item in asr)
        )

        summaries = hydrator.get_summary_evidence(("L21_V002",))
        self.assertEqual(tuple(item.video_id for item in summaries), ("L21_V002",))

    def test_empty_success_is_distinct_from_backend_failure(self) -> None:
        hydrator = self.fixture.evidence_hydrator()
        self.assertEqual(hydrator.get_ocr_evidence(("L21_V004_001",)), ())

        failed = self.fixture.evidence_hydrator(
            behaviors={
                "ocr": AdvancedFakeBehavior(
                    ResourceUnavailableError("simulated OCR outage")
                )
            }
        )
        with self.assertRaises(ResourceUnavailableError):
            failed.get_ocr_evidence(("L21_V004_001",))

    def test_missing_image_is_distinct_from_resolver_unavailable(self) -> None:
        resolver = self.fixture.image_resolver()
        self.assertEqual(
            dict(resolver.resolve_images((self.fixture.missing_image_frame_id,))),
            {},
        )
        failed = self.fixture.image_resolver(
            behavior=AdvancedFakeBehavior(
                ResourceUnavailableError("simulated resolver outage")
            )
        )
        with self.assertRaises(ResourceUnavailableError):
            failed.resolve_images((self.fixture.missing_image_frame_id,))

    def test_fake_vlm_modes_cover_success_insufficient_timeout_unavailable_and_malformed(
        self,
    ) -> None:
        request = self._vlm_request()
        answered = self.fixture.vlm(FakeVLMMode.ANSWERED).answer(request)
        self.assertIsInstance(answered, VLMResponse)
        self.assertEqual(answered.status, VLMResponseStatus.ANSWERED)
        self.assertEqual(
            answered.evidence_ids,
            self.fixture.expected_vqa_answer_evidence_ids,
        )

        insufficient = self.fixture.vlm(FakeVLMMode.INSUFFICIENT).answer(request)
        self.assertEqual(
            insufficient.status, VLMResponseStatus.INSUFFICIENT_EVIDENCE
        )
        self.assertIsNone(insufficient.answer)

        with self.assertRaises(BranchTimeoutError):
            self.fixture.vlm(FakeVLMMode.TIMEOUT).answer(request)
        with self.assertRaises(ResourceUnavailableError):
            self.fixture.vlm(FakeVLMMode.UNAVAILABLE).answer(request)
        malformed = self.fixture.vlm(FakeVLMMode.MALFORMED).answer(request)
        self.assertNotIsInstance(malformed, VLMResponse)

    def test_fake_vlm_explicit_grounding_override_supports_image_only_request(
        self,
    ) -> None:
        image_id = self.fixture.expected_vqa_selected_evidence_ids[0]
        request = self._vlm_request((image_id,))
        response = self.fixture.vlm(
            grounded_evidence_ids=(image_id,),
        ).answer(request)
        self.assertEqual(response.evidence_ids, (image_id,))

    def test_fake_vlm_rejects_explicit_empty_grounding_without_fallback(
        self,
    ) -> None:
        request = self._vlm_request()
        with self.assertRaises(ContractMismatchError):
            self.fixture.vlm(grounded_evidence_ids=()).answer(request)

    def test_fake_vlm_rejects_unknown_grounding_when_processing_request(
        self,
    ) -> None:
        request = self._vlm_request()
        with self.assertRaises(ContractMismatchError):
            self.fixture.vlm(
                grounded_evidence_ids=("image:unknown",),
            ).answer(request)

    def test_call_logs_are_bounded_and_do_not_expose_vectors_secrets_or_local_paths(
        self,
    ) -> None:
        encoder = self.fixture.text_encoder()
        encoder.encode_texts(
            tuple(event.text for event in self.fixture.trake_query.events)
        )
        corpus = self.fixture.visual_corpus()
        corpus.iter_ordered_frame_embedding_batches("L21_V001", 2)
        hydrator = self.fixture.evidence_hydrator()
        hydrator.get_summary_evidence(("L21_V001",))
        resolver = self.fixture.image_resolver()
        resolver.resolve_images((next(iter(self.fixture.images_by_frame_id)),))
        vlm = self.fixture.vlm()
        vlm.answer(self._vlm_request())

        logs = repr(
            (
                encoder.calls,
                corpus.calls,
                hydrator.calls,
                resolver.calls,
                vlm.calls,
            )
        ).lower()
        for forbidden in ("vector=", "api_key", "secret", "file://", "c:\\"):
            self.assertNotIn(forbidden, logs)


if __name__ == "__main__":
    unittest.main()
