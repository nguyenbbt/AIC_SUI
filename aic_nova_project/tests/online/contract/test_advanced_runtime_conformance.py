from __future__ import annotations

import builtins
import math
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.domain.vqa import VLMRequest
from online.ports.encoders import TextEncoderPort
from online.ports.evidence import EvidenceHydrationPort
from online.ports.images import ImageResolverPort
from online.ports.metadata import MetadataReaderPort
from online.ports.visual_corpus import VisualCorpusPort
from online.ports.vlm import VLMPort
from online.testing import (
    AdvancedRuntimeState,
    build_advanced_runtime_bundle,
    build_happy_path_advanced_runtime_bundle,
)


class AdvancedRuntimeConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = build_happy_path_advanced_runtime_bundle(request_id="contract-a")
        self.fixture = self.bundle.fixture

    def tearDown(self) -> None:
        if not self.bundle.closed:
            self.bundle.close()

    def _vlm_request(self, request_id: str = "vqa-contract-a") -> VLMRequest:
        records = (
            tuple(self.fixture.images_by_frame_id.values())
            + self.fixture.ocr_evidence
            + self.fixture.asr_evidence
            + self.fixture.summary_evidence
        )
        by_id = {record.evidence_id: record for record in records}
        return VLMRequest(
            request_id=request_id,
            question=self.fixture.vqa_question,
            evidence=tuple(
                by_id[evidence_id]
                for evidence_id in self.fixture.expected_vqa_answer_evidence_ids
            ),
        )

    def test_public_bundle_components_conform_to_frozen_ports(self) -> None:
        self.assertIsInstance(self.bundle.text_encoder, TextEncoderPort)
        self.assertIsInstance(self.bundle.visual_corpus, VisualCorpusPort)
        self.assertIsInstance(self.bundle.metadata_reader, MetadataReaderPort)
        self.assertIsInstance(self.bundle.evidence_hydrator, EvidenceHydrationPort)
        self.assertIsInstance(self.bundle.image_resolver, ImageResolverPort)
        self.assertIsInstance(self.bundle.vlm, VLMPort)

    def test_happy_path_preserves_ids_and_returns_defensive_snapshots(self) -> None:
        frame = self.fixture.visual_frames_by_video["L21_V001"][0]
        batches = self.bundle.visual_corpus.iter_ordered_frame_embedding_batches(
            "L21_V001",
            2,
        )
        self.assertEqual(batches[0][0].frame_id, frame.frame_id)
        self.assertEqual(batches[0][0].video_id, frame.video_id)
        metadata = self.bundle.metadata_reader.get_frames_by_ids((frame.frame_id,))
        self.assertEqual(metadata[frame.frame_id].frame_id, frame.frame_id)
        with self.assertRaises(TypeError):
            metadata[frame.frame_id] = frame  # type: ignore[index]

        image_id = next(iter(self.fixture.images_by_frame_id))
        images = self.bundle.image_resolver.resolve_images((image_id,))
        self.assertEqual(images[image_id].frame_id, image_id)
        with self.assertRaises(TypeError):
            images[image_id] = self.fixture.images_by_frame_id[image_id]  # type: ignore[index]

    def test_happy_path_preserves_complete_records_and_provenance(self) -> None:
        for video_id, expected_frames in self.fixture.visual_frames_by_video.items():
            batches = self.bundle.visual_corpus.iter_ordered_frame_embedding_batches(
                video_id,
                2,
            )
            self.assertEqual(
                tuple(frame for batch in batches for frame in batch),
                expected_frames,
            )

        frame_ids = tuple(frame.frame_id for frame in self.fixture.frame_metadata)
        metadata = self.bundle.metadata_reader.get_frames_by_ids(frame_ids)
        self.assertEqual(
            tuple(metadata[frame_id] for frame_id in frame_ids),
            self.fixture.frame_metadata,
        )

        ocr_frame_ids = tuple(record.frame_id for record in self.fixture.ocr_evidence)
        self.assertEqual(
            self.bundle.evidence_hydrator.get_ocr_evidence(ocr_frame_ids),
            self.fixture.ocr_evidence,
        )
        video_ids = tuple(record.video_id for record in self.fixture.summary_evidence)
        self.assertEqual(
            self.bundle.evidence_hydrator.get_summary_evidence(video_ids),
            self.fixture.summary_evidence,
        )
        for expected in self.fixture.asr_evidence:
            self.assertEqual(
                self.bundle.evidence_hydrator.get_asr_evidence(
                    expected.video_id,
                    expected.start_time_sec,
                    expected.end_time_sec,
                ),
                (expected,),
            )

        image_ids = tuple(self.fixture.images_by_frame_id)
        self.assertEqual(
            dict(self.bundle.image_resolver.resolve_images(image_ids)),
            dict(self.fixture.images_by_frame_id),
        )
        self.assertEqual(
            self.bundle.vlm.answer(self._vlm_request()).evidence_ids,
            self.fixture.expected_vqa_answer_evidence_ids,
        )

    def test_each_resource_distinguishes_empty_timeout_unavailable_and_invalid(self) -> None:
        encoder_empty = build_advanced_runtime_bundle(
            encoder_state=AdvancedRuntimeState.EMPTY
        )
        self.addCleanup(encoder_empty.close)
        self.assertEqual(encoder_empty.text_encoder.encode_texts(("event",)), ())

        corpus_empty = build_advanced_runtime_bundle(
            visual_state=AdvancedRuntimeState.EMPTY
        )
        self.addCleanup(corpus_empty.close)
        self.assertEqual(corpus_empty.visual_corpus.list_video_ids(), ())

        metadata_empty = build_advanced_runtime_bundle(
            metadata_state=AdvancedRuntimeState.EMPTY
        )
        self.addCleanup(metadata_empty.close)
        self.assertEqual(
            metadata_empty.metadata_reader.get_frames_by_ids(("frame",)),
            {},
        )

        empty = build_advanced_runtime_bundle(image_state=AdvancedRuntimeState.EMPTY)
        self.addCleanup(empty.close)
        self.assertEqual(empty.image_resolver.resolve_images(("unknown-frame",)), {})

        for state, error_type in (
            (AdvancedRuntimeState.TIMEOUT, BranchTimeoutError),
            (AdvancedRuntimeState.UNAVAILABLE, ResourceUnavailableError),
            (AdvancedRuntimeState.INVALID_REFERENCE, ContractMismatchError),
        ):
            bundle = build_advanced_runtime_bundle(image_state=state)
            self.addCleanup(bundle.close)
            with self.subTest(state=state):
                with self.assertRaises(error_type):
                    bundle.image_resolver.resolve_images(("frame",))

        evidence_empty = build_advanced_runtime_bundle(
            evidence_state=AdvancedRuntimeState.EMPTY
        )
        self.addCleanup(evidence_empty.close)
        self.assertEqual(evidence_empty.evidence_hydrator.get_ocr_evidence(("frame",)), ())

        vlm_empty = build_advanced_runtime_bundle(vlm_state=AdvancedRuntimeState.EMPTY)
        self.addCleanup(vlm_empty.close)
        response = vlm_empty.vlm.answer(self._vlm_request())
        self.assertEqual(response.status.value, "insufficient_evidence")

    def test_failure_states_are_configurable_for_every_resource(self) -> None:
        fixture = self.fixture
        known_frame = fixture.frame_metadata[0]
        operations = (
            (
                "encoder_state",
                lambda bundle: bundle.text_encoder.encode_texts(
                    (fixture.trake_query.events[0].text,)
                ),
            ),
            (
                "visual_state",
                lambda bundle: bundle.visual_corpus.list_video_ids(),
            ),
            (
                "metadata_state",
                lambda bundle: bundle.metadata_reader.get_frames_by_ids(
                    (known_frame.frame_id,)
                ),
            ),
            (
                "ocr_state",
                lambda bundle: bundle.evidence_hydrator.get_ocr_evidence(
                    (fixture.ocr_evidence[0].frame_id,)
                ),
            ),
            (
                "asr_state",
                lambda bundle: bundle.evidence_hydrator.get_asr_evidence(
                    fixture.asr_evidence[0].video_id,
                    fixture.asr_evidence[0].start_time_sec,
                    fixture.asr_evidence[0].end_time_sec,
                ),
            ),
            (
                "summary_state",
                lambda bundle: bundle.evidence_hydrator.get_summary_evidence(
                    (fixture.summary_evidence[0].video_id,)
                ),
            ),
            (
                "image_state",
                lambda bundle: bundle.image_resolver.resolve_images(
                    (next(iter(fixture.images_by_frame_id)),)
                ),
            ),
            (
                "vlm_state",
                lambda bundle: bundle.vlm.answer(self._vlm_request()),
            ),
        )
        for field_name, operation in operations:
            for state, error_type in (
                (AdvancedRuntimeState.TIMEOUT, BranchTimeoutError),
                (AdvancedRuntimeState.UNAVAILABLE, ResourceUnavailableError),
                (AdvancedRuntimeState.INVALID_REFERENCE, ContractMismatchError),
            ):
                bundle = build_advanced_runtime_bundle(**{field_name: state})
                self.addCleanup(bundle.close)
                with self.subTest(resource=field_name, state=state):
                    with self.assertRaises(error_type):
                        operation(bundle)

    def test_invalid_reference_state_is_safe_and_does_not_echo_input(self) -> None:
        secret_reference = "https://provider.invalid/image?token=do-not-log"
        bundle = build_advanced_runtime_bundle(
            image_state=AdvancedRuntimeState.INVALID_REFERENCE,
            request_id="safe-request",
        )
        self.addCleanup(bundle.close)
        with self.assertRaises(ContractMismatchError) as caught:
            bundle.image_resolver.resolve_images((secret_reference,))
        self.assertNotIn("do-not-log", str(caught.exception))
        self.assertNotIn(secret_reference, repr(bundle.calls))

    def test_caller_input_is_validated_before_every_configured_state(self) -> None:
        for state in AdvancedRuntimeState:
            bundle = build_advanced_runtime_bundle(image_state=state)
            self.addCleanup(bundle.close)
            with self.subTest(state=state):
                with self.assertRaises(InvalidQueryError):
                    bundle.image_resolver.resolve_images("not-a-sequence")
                self.assertEqual(bundle.calls, ())

    def test_safe_retryable_semantic_survives_vlm_wrapper(self) -> None:
        class RetryableVLM:
            def answer(self, request):
                raise ResourceUnavailableError(
                    "provider payload must not escape",
                    details={
                        "retryable": True,
                        "provider_payload": "secret",
                    },
                )

        bundle = build_advanced_runtime_bundle(vlm=RetryableVLM())
        self.addCleanup(bundle.close)
        with self.assertRaises(ResourceUnavailableError) as caught:
            bundle.vlm.answer(self._vlm_request())
        self.assertIs(caught.exception.details["retryable"], True)
        self.assertNotIn("provider payload", str(caught.exception).lower())
        self.assertNotIn("secret", repr(caught.exception.details).lower())

    def test_malformed_dependency_output_uses_contract_error_taxonomy(self) -> None:
        class MalformedImageResolver:
            def resolve_images(self, frame_ids):
                return None

        bundle = build_advanced_runtime_bundle(
            image_resolver=MalformedImageResolver()
        )
        self.addCleanup(bundle.close)
        with self.assertRaises(ContractMismatchError):
            bundle.image_resolver.resolve_images(("L21_V001_005",))

    def test_invalid_image_and_evidence_provenance_fail_safe(self) -> None:
        wrong_ocr = self.fixture.ocr_evidence[0]
        wrong_image = next(iter(self.fixture.images_by_frame_id.values()))

        class WrongEvidenceHydrator:
            def get_ocr_evidence(self, frame_ids):
                return (wrong_ocr,)

            def get_asr_evidence(self, video_id, start_sec, end_sec):
                return ()

            def get_summary_evidence(self, video_ids):
                return ()

        class WrongImageResolver:
            def resolve_images(self, frame_ids):
                return {wrong_image.frame_id: wrong_image}

        evidence_bundle = build_advanced_runtime_bundle(
            evidence_hydrator=WrongEvidenceHydrator()
        )
        image_bundle = build_advanced_runtime_bundle(
            image_resolver=WrongImageResolver()
        )
        self.addCleanup(evidence_bundle.close)
        self.addCleanup(image_bundle.close)
        with self.assertRaises(ContractMismatchError):
            evidence_bundle.evidence_hydrator.get_ocr_evidence(
                ("L21_V004_001",)
            )
        with self.assertRaises(ContractMismatchError):
            image_bundle.image_resolver.resolve_images(("L21_V004_001",))

    def test_encoder_dimension_is_guarded_and_sanitized(self) -> None:
        class LeakyEncoder:
            fail = False

            @property
            def dimension(self):
                if self.fail:
                    raise ResourceUnavailableError(
                        "provider secret token=must-not-escape"
                    )
                return 4

            def encode_texts(self, texts):
                return tuple((1.0, 0.0, 0.0, 0.0) for _ in texts)

        delegate = LeakyEncoder()
        bundle = build_advanced_runtime_bundle(text_encoder=delegate)
        self.addCleanup(bundle.close)
        delegate.fail = True
        with self.assertRaises(ResourceUnavailableError) as caught:
            _ = bundle.text_encoder.dimension
        self.assertNotIn("must-not-escape", str(caught.exception))
        bundle.close()
        with self.assertRaises(ResourceUnavailableError):
            _ = bundle.text_encoder.dimension

    def test_malformed_vlm_output_is_preserved_but_defensively_frozen(self) -> None:
        mutable_result = {
            "status": "answered",
            "evidence_ids": ["image:unknown"],
        }

        class MalformedVLM:
            def answer(self, request):
                return mutable_result

        bundle = build_advanced_runtime_bundle(vlm=MalformedVLM())
        self.addCleanup(bundle.close)
        result = bundle.vlm.answer(self._vlm_request())
        self.assertEqual(result["evidence_ids"], ("image:unknown",))
        mutable_result["evidence_ids"].append("image:later")
        self.assertEqual(result["evidence_ids"], ("image:unknown",))
        with self.assertRaises(TypeError):
            result["status"] = "changed"

    def test_determinism_and_instance_isolation(self) -> None:
        first = build_advanced_runtime_bundle(request_id="first")
        second = build_advanced_runtime_bundle(request_id="second")
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.assertEqual(first.fixture, second.fixture)
        self.assertEqual(
            first.visual_corpus.list_video_ids(),
            second.visual_corpus.list_video_ids(),
        )
        self.assertEqual(first.calls[0].sequence, 1)
        self.assertEqual(second.calls[0].sequence, 1)
        self.assertEqual(first.calls[0].request_id, "first")
        self.assertEqual(second.calls[0].request_id, "second")

    def test_request_scoped_bundles_keep_logs_and_responses_isolated(self) -> None:
        first = self.bundle.for_request("request-one")
        second = self.bundle.for_request("request-two")
        self.addCleanup(first.close)
        self.addCleanup(second.close)

        def run(bundle):
            return bundle.visual_corpus.iter_ordered_frame_embedding_batches("L21_V001", 2)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_result, second_result = tuple(
                executor.map(run, (first, second))
            )
        self.assertEqual(first_result, second_result)
        self.assertEqual(
            tuple(call.request_id for call in first.calls),
            ("request-one",),
        )
        self.assertEqual(
            tuple(call.request_id for call in second.calls),
            ("request-two",),
        )
        self.assertEqual(first.calls[0].sequence, 1)
        self.assertEqual(second.calls[0].sequence, 1)

    def test_call_log_is_ordered_bounded_and_safe(self) -> None:
        self.bundle.visual_corpus.list_video_ids()
        self.bundle.visual_corpus.iter_ordered_frame_embedding_batches("L21_V001", 2)
        self.bundle.image_resolver.resolve_images(
            (next(iter(self.fixture.images_by_frame_id)),)
        )
        self.bundle.vlm.answer(self._vlm_request())
        calls = self.bundle.call_log.snapshot()
        self.assertEqual(tuple(call.sequence for call in calls), tuple(range(1, 5)))
        self.assertEqual(calls[0].request_id, "contract-a")
        self.assertEqual(calls[-1].request_id, "vqa-contract-a")
        self.assertTrue(all(len(call.identifiers) <= 16 for call in calls))
        rendered = repr(calls).lower()
        for forbidden in ("vector=", "token=do-not-log", "file://", "c:\\"):
            self.assertNotIn(forbidden, rendered)

    def test_safe_happy_fakes_perform_no_filesystem_or_socket_io(self) -> None:
        with patch.object(
            builtins,
            "open",
            side_effect=AssertionError("filesystem access is forbidden"),
        ), patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("network access is forbidden"),
        ):
            self.bundle.visual_corpus.list_video_ids()
            self.bundle.image_resolver.resolve_images(
                (next(iter(self.fixture.images_by_frame_id)),)
            )
            self.bundle.vlm.answer(self._vlm_request())

    def test_custom_vlm_request_id_is_preserved_at_boundary(self) -> None:
        request = self._vlm_request(request_id="request/with-safe-boundary-id")
        response = self.bundle.vlm.answer(request)
        self.assertEqual(
            response.evidence_ids,
            self.fixture.expected_vqa_answer_evidence_ids,
        )
        self.assertEqual(
            self.bundle.calls[-1].request_id,
            "request/with-safe-boundary-id",
        )

    def test_path_like_request_id_is_correlated_without_path_disclosure(self) -> None:
        path_request_id = r"C:\Users\example\private-evidence.md"
        bundle = build_advanced_runtime_bundle(request_id=path_request_id)
        self.addCleanup(bundle.close)
        bundle.visual_corpus.list_video_ids()
        logged_id = bundle.calls[0].request_id
        self.assertTrue(logged_id.startswith("[redacted-request-id:"))
        self.assertNotIn("Users", logged_id)
        self.assertNotIn(path_request_id, repr(bundle.calls))

    def test_norms_and_dimensions_remain_unchanged(self) -> None:
        texts = tuple(event.text for event in self.fixture.trake_query.events)
        vectors = self.bundle.text_encoder.encode_texts(texts)
        self.assertEqual(len(vectors), len(texts))
        self.assertEqual({len(vector) for vector in vectors}, {4})
        self.assertTrue(
            all(
                math.isclose(
                    math.sqrt(sum(value * value for value in vector)),
                    1.0,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
                for vector in vectors
            )
        )

    def test_lazy_port_iteration_stays_active_and_sanitizes_errors(self) -> None:
        frame = self.fixture.visual_frames_by_video["L21_V001"][0]

        class LazyCorpus:
            def list_video_ids(self):
                return ("L21_V001",)

            def iter_ordered_frame_embedding_batches(self, video_id, batch_size):
                def batches():
                    yield (frame,)
                    raise ResourceUnavailableError("secret token=must-not-escape")

                return batches()

        bundle = build_advanced_runtime_bundle(visual_corpus=LazyCorpus())
        self.addCleanup(bundle.close)
        with self.assertRaises(ResourceUnavailableError) as caught:
            bundle.visual_corpus.iter_ordered_frame_embedding_batches("L21_V001", 1)
        self.assertNotIn("must-not-escape", str(caught.exception))
        self.assertEqual(bundle.trake_lifecycle.active_count, 0)
        self.assertFalse(bundle.closed)


if __name__ == "__main__":
    unittest.main()
