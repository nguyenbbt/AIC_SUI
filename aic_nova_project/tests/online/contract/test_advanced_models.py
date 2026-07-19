from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from online.domain.trake import (
    DANTEPolicy,
    TRAKEDiagnostics,
    TRAKEEvent,
    TRAKEFrameMatch,
    TRAKEQuery,
    TRAKEVideoResult,
)
from online.domain.vqa import (
    ASREvidence,
    ImageEvidence,
    OCREvidence,
    SummaryEvidence,
    VLMConfidence,
    VLMRequest,
    VLMResponse,
    VLMResponseStatus,
    VQAAnswerType,
    VQADiagnostics,
    VQAEvidenceBudget,
    VQAQuestion,
    VQAResult,
)


def _events() -> tuple[TRAKEEvent, ...]:
    return (
        TRAKEEvent(event_id="e1", text="Một người bước vào phòng"),
        TRAKEEvent(event_id="e2", text="Người đó ngồi xuống"),
    )


def _match(
    event_id: str,
    *,
    local_index: int,
    frame_id: str,
    shot_id: int,
    video_id: str = "V001",
) -> TRAKEFrameMatch:
    return TRAKEFrameMatch(
        event_id=event_id,
        frame_id=frame_id,
        video_id=video_id,
        shot_id=shot_id,
        local_index=local_index,
        timestamp_sec=float(local_index),
        similarity_score=0.75,
    )


def _image() -> ImageEvidence:
    return ImageEvidence(
        evidence_id="image:V001_00000_015",
        video_id="V001",
        frame_id="V001_00000_015",
        shot_id=0,
        timestamp_sec=1.5,
        image_reference="fixture://images/V001_00000_015",
    )


class TRAKEAdvancedModelTests(unittest.TestCase):
    def test_success_serialization_round_trip(self) -> None:
        query = TRAKEQuery(query_id="trake-1", events=_events(), top_k_videos=3)
        result = TRAKEVideoResult(
            video_id="V001",
            score=1.49,
            event_ids=("e1", "e2"),
            sequence=(
                _match("e1", local_index=0, frame_id="V001_00000_015", shot_id=0),
                _match("e2", local_index=3, frame_id="V001_00001_050", shot_id=1),
            ),
        )
        diagnostics = TRAKEDiagnostics(
            policy_version=query.policy.policy_version,
            lambda_penalty=query.policy.lambda_penalty,
            event_count=2,
            video_count=2,
            frame_count=12,
        )

        restored_query = TRAKEQuery.model_validate_json(query.model_dump_json())
        restored_result = TRAKEVideoResult.model_validate_json(result.model_dump_json())
        self.assertEqual(restored_query, query)
        self.assertEqual(restored_result, result)
        self.assertEqual(json.loads(diagnostics.model_dump_json())["event_count"], 2)
        self.assertNotIn("vector", result.model_dump())

    def test_empty_one_or_duplicate_events_are_rejected(self) -> None:
        for events in (
            (),
            (_events()[0],),
            (_events()[0], TRAKEEvent(event_id="e1", text="Khác")),
        ):
            with self.assertRaises(ValidationError):
                TRAKEQuery(query_id="trake-1", events=events)

    def test_top_k_and_lambda_numeric_contract(self) -> None:
        for invalid_top_k in (0, -1, True):
            with self.assertRaises(ValidationError):
                TRAKEQuery(query_id="trake-1", events=_events(), top_k_videos=invalid_top_k)

        self.assertEqual(DANTEPolicy(lambda_penalty=0.001).lambda_penalty, 0.001)
        self.assertEqual(DANTEPolicy(lambda_penalty=0.01).lambda_penalty, 0.01)
        for invalid_lambda in (0.0009, 0.0101, float("nan"), float("inf"), True):
            with self.assertRaises(ValidationError):
                DANTEPolicy(lambda_penalty=invalid_lambda)

    def test_sequence_rejects_cross_video_non_increasing_missing_or_duplicate_event(self) -> None:
        first = _match("e1", local_index=1, frame_id="V001_00000_015", shot_id=0)
        cases = (
            (
                first,
                _match(
                    "e2",
                    local_index=2,
                    frame_id="V002_00001_050",
                    shot_id=1,
                    video_id="V002",
                ),
            ),
            (
                first,
                _match("e2", local_index=1, frame_id="V001_00001_050", shot_id=1),
            ),
            (
                first,
                _match("e3", local_index=2, frame_id="V001_00001_050", shot_id=1),
            ),
        )
        for sequence in cases:
            with self.assertRaises(ValidationError):
                TRAKEVideoResult(
                    video_id="V001",
                    score=1.0,
                    event_ids=("e1", "e2"),
                    sequence=sequence,
                )

    def test_match_rejects_non_canonical_identity_non_finite_score_and_boolean(self) -> None:
        base = {
            "event_id": "e1",
            "frame_id": "V001_00000_015",
            "video_id": "V001",
            "shot_id": 0,
            "local_index": 0,
            "timestamp_sec": 1.5,
            "similarity_score": 0.7,
        }
        for field, value in (
            ("frame_id", "shot_00000_pos_015"),
            ("video_id", "V002"),
            ("shot_id", 1),
            ("similarity_score", float("nan")),
            ("similarity_score", True),
            ("local_index", True),
        ):
            with self.assertRaises((ValidationError, ValueError)):
                TRAKEFrameMatch.model_validate({**base, field: value})

    def test_models_reject_extra_fields_and_are_frozen(self) -> None:
        with self.assertRaises(ValidationError):
            TRAKEEvent(event_id="e1", text="event", sdk_row=object())  # type: ignore[call-arg]
        event = _events()[0]
        with self.assertRaises(ValidationError):
            event.text = "mutated"  # type: ignore[misc]


class VQAAdvancedModelTests(unittest.TestCase):
    def test_budget_defaults_match_dd_030_and_round_trip(self) -> None:
        budget = VQAEvidenceBudget()
        self.assertEqual(
            budget.model_dump(),
            {
                "max_videos": 3,
                "max_primary_frames_per_video": 3,
                "max_primary_frames_total": 8,
                "max_images_total": 12,
                "max_ocr_chars": 2_000,
                "max_asr_chars": 4_000,
                "max_summary_chars_per_video": 800,
                "max_summary_chars_total": 2_400,
                "max_text_chars_total": 8_000,
                "asr_window_sec": 5.0,
            },
        )
        self.assertEqual(
            VQAEvidenceBudget.model_validate_json(budget.model_dump_json()),
            budget,
        )
        for override in (
            {"max_primary_frames_total": 13},
            {"max_videos": 9},
            {"max_summary_chars_total": 8_001},
            {"asr_window_sec": float("inf")},
        ):
            with self.assertRaises(ValidationError):
                VQAEvidenceBudget(**override)

    def test_answered_and_insufficient_response_rules(self) -> None:
        answered = VLMResponse(
            status=VLMResponseStatus.ANSWERED,
            answer="Có",
            answer_type=VQAAnswerType.YES_NO,
            confidence=VLMConfidence.HIGH,
            evidence_ids=("image:V001_00000_015",),
        )
        self.assertEqual(answered.answer, "Có")
        insufficient = VLMResponse(
            status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
            answer=" ",
            answer_type=VQAAnswerType.SHORT_TEXT,
            confidence=VLMConfidence.LOW,
            evidence_ids=(),
        )
        self.assertIsNone(insufficient.answer)

        invalid_payloads = (
            {
                "status": "answered",
                "answer": None,
                "answer_type": "short_text",
                "confidence": "low",
                "evidence_ids": ("image:1",),
            },
            {
                "status": "answered",
                "answer": "Có",
                "answer_type": "yes_no",
                "confidence": "high",
                "evidence_ids": (),
            },
            {
                "status": "insufficient_evidence",
                "answer": "Bịa",
                "answer_type": "short_text",
                "confidence": "low",
                "evidence_ids": (),
            },
        )
        for payload in invalid_payloads:
            with self.assertRaises(ValidationError):
                VLMResponse.model_validate(payload)

    def test_evidence_types_preserve_candidate_level(self) -> None:
        image = _image()
        ocr = OCREvidence(
            evidence_id="ocr:V001_00000_015",
            video_id="V001",
            frame_id="V001_00000_015",
            text="BẢNG HIỆU",
        )
        asr = ASREvidence(
            evidence_id="asr:V001:interval_1",
            video_id="V001",
            interval_id="interval_1",
            start_time_sec=1.0,
            end_time_sec=4.0,
            text="Xin chào",
        )
        summary = SummaryEvidence(
            evidence_id="summary:V001",
            video_id="V001",
            text="Một người đi vào phòng.",
        )
        evidence = (image, ocr, asr, summary)
        restored = VLMRequest.model_validate_json(
            VLMRequest(
                request_id="vlm-1",
                question=VQAQuestion(
                    question_id="vqa-1",
                    question="Người đó làm gì?",
                    answer_type=VQAAnswerType.SHORT_TEXT,
                ),
                evidence=evidence,
            ).model_dump_json()
        )
        self.assertEqual(tuple(item.evidence_type.value for item in restored.evidence), (
            "image",
            "ocr",
            "asr",
            "summary",
        ))
        self.assertFalse(hasattr(restored.evidence[-1], "frame_id"))
        self.assertEqual(restored.evidence[2].interval_id, "interval_1")  # type: ignore[union-attr]

    def test_asr_interval_and_summary_level_validation(self) -> None:
        with self.assertRaises(ValidationError):
            ASREvidence(
                evidence_id="asr:1",
                video_id="V001",
                interval_id="interval_1",
                start_time_sec=4.0,
                end_time_sec=1.0,
                text="text",
            )
        with self.assertRaises(ValidationError):
            SummaryEvidence(
                evidence_id="summary:V001",
                video_id="V001",
                text="summary",
                frame_id="V001_00000_015",  # type: ignore[call-arg]
            )

    def test_public_evidence_rejects_absolute_paths_and_wrong_frame_identity(self) -> None:
        for reference in (
            "C:\\private\\V001.webp",
            "/srv/private/V001.webp",
            "\\\\server\\private\\V001.webp",
            "https://example.test/V001.webp?api_key=secret",
            "https://user:password@example.test/V001.webp",
        ):
            with self.assertRaises(ValidationError):
                ImageEvidence(
                    evidence_id="image:1",
                    video_id="V001",
                    frame_id="V001_00000_015",
                    shot_id=0,
                    timestamp_sec=1.5,
                    image_reference=reference,
                )
        with self.assertRaises(ValidationError):
            OCREvidence(
                evidence_id="ocr:1",
                video_id="V002",
                frame_id="V001_00000_015",
                text="text",
            )

    def test_result_enforces_grounded_evidence_ids(self) -> None:
        image = _image()
        response = VLMResponse(
            status="answered",
            answer="Một người",
            answer_type="short_text",
            confidence="medium",
            evidence_ids=(image.evidence_id,),
        )
        result = VQAResult(
            question_id="vqa-1",
            response=response,
            evidence=(image,),
            diagnostics=VQADiagnostics(selected_image_count=1),
        )
        self.assertEqual(
            VQAResult.model_validate_json(result.model_dump_json()),
            result,
        )
        with self.assertRaises(ValidationError):
            VQAResult(
                question_id="vqa-1",
                response=response.model_copy(
                    update={"evidence_ids": ("image:unknown",)}
                ),
                evidence=(image,),
            )

    def test_extra_fields_boolean_caps_and_frozen_behavior_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            VQAQuestion(
                question_id="vqa-1",
                question="Có ai không?",
                answer_type="yes_no",
                api_key="secret",  # type: ignore[call-arg]
            )
        with self.assertRaises(ValidationError):
            VQAEvidenceBudget(max_videos=True)
        budget = VQAEvidenceBudget()
        with self.assertRaises(ValidationError):
            budget.max_videos = 4  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
