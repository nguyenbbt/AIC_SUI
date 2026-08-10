from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    FrameCandidate,
    FusedFrameCandidate,
    ObjectDetection,
    VideoCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.domain.identifiers import (
    parse_canonical_frame_id,
    validate_canonical_frame_id,
)
from online.domain.errors import ResourceUnavailableError
from online.domain.query import ObjectConstraint
from online.domain.trake import TRAKEFrameMatch
from online.domain.vqa import ImageEvidence
from online.ports.records import FrameMetadata, FrameSearchHit
from online.ports.visual_corpus import OrderedVisualFrame
from online.testing.organizer_fixtures import build_self_indexed_frame_metadata


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "candidates.json"


class DomainContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_candidate_json_round_trip_preserves_enums_and_provenance(self) -> None:
        for model, key in (
            (FrameCandidate, "frame"),
            (ASRIntervalCandidate, "asr_interval"),
            (VideoCandidate, "video"),
        ):
            candidate = model.model_validate(self.fixture[key])
            restored = model.model_validate_json(candidate.model_dump_json())
            self.assertEqual(candidate, restored)
            self.assertIsInstance(restored.provenance.branch, RetrievalBranch)

    def test_rejects_empty_id_negative_rank_unknown_field_and_non_finite_score(self) -> None:
        payload = dict(self.fixture["frame"])
        for field, value in (
            ("frame_id", " "),
            ("rank", 0),
            ("timestamp_sec", -1),
            ("raw_score", float("nan")),
        ):
            invalid = {**payload, field: value}
            with self.assertRaises(ValidationError):
                FrameCandidate.model_validate(invalid)
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**payload, "unknown": True})

    def test_interval_end_must_not_precede_start(self) -> None:
        payload = {**self.fixture["asr_interval"], "start_time_sec": 7, "end_time_sec": 6}
        with self.assertRaises(ValidationError):
            ASRIntervalCandidate.model_validate(payload)

    def test_interval_id_is_unpadded_non_negative_decimal(self) -> None:
        payload = self.fixture["asr_interval"]
        for invalid in ("", "00", "01", "-1", "interval-1", " 1", "1 "):
            with self.assertRaises(ValidationError):
                ASRIntervalCandidate.model_validate({**payload, "interval_id": invalid})

    def test_object_constraint_rejects_negative_count(self) -> None:
        with self.assertRaises(ValidationError):
            ObjectConstraint(
                label="person",
                count_operator="gte",
                count=-1,
                min_confidence=0.5,
            )

    def test_object_detection_uses_self_indexed_schema_and_normalized_bbox(self) -> None:
        detection = ObjectDetection(
            label="person",
            confidence=0.95,
            x_min=0.10,
            y_min=0.20,
            x_max=0.60,
            y_max=0.90,
            model_source="yolo",
        )
        self.assertEqual(
            set(ObjectDetection.model_fields),
            {
                "label",
                "confidence",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "model_source",
            },
        )
        self.assertEqual(
            ObjectDetection.model_validate_json(detection.model_dump_json()),
            detection,
        )
        without_source = detection.model_copy(update={"model_source": None})
        self.assertIsNone(without_source.model_source)

    def test_object_detection_rejects_legacy_or_invalid_records(self) -> None:
        base = {
            "label": "person",
            "confidence": 0.95,
            "x_min": 0.10,
            "y_min": 0.20,
            "x_max": 0.60,
            "y_max": 0.90,
            "model_source": "yolo",
        }
        for field, value in (
            ("label", "Person"),
            ("label", " PERSON "),
            ("confidence", 1.01),
            ("x_min", -0.01),
            ("y_min", 1.01),
            ("x_max", 1.01),
            ("y_max", -0.01),
            ("x_min", 0.70),
            ("y_min", 0.95),
            ("x_max", 0.10),
            ("y_max", 0.20),
            ("model_source", " "),
        ):
            with self.assertRaises(ValidationError):
                ObjectDetection.model_validate({**base, field: value})
        with self.assertRaises(ValidationError):
            ObjectDetection.model_validate({**base, "label_display": "Person"})

    def test_error_diagnostics_redact_secrets_and_vectors(self) -> None:
        error = ResourceUnavailableError(
            "backend unavailable",
            details={
                "uri": "https://user:password@example.test:9200",
                "token": "secret-token",
                "vector": [1.0, 2.0],
                "operation": "search",
            },
        )
        safe = error.to_safe_dict()
        self.assertNotIn("password", str(safe))
        self.assertNotIn("secret-token", str(safe))
        self.assertEqual(safe["details"]["operation"], "search")

        signed = ResourceUnavailableError(
            "backend unavailable",
            details={
                "uri": "https://example.test/private/path?token=secret-value#fragment"
            },
        ).to_safe_dict()
        self.assertEqual(signed["details"]["uri"], "https://example.test")
        self.assertNotIn("secret-value", str(signed))

    def test_models_are_frozen(self) -> None:
        candidate = FrameCandidate.model_validate(self.fixture["frame"])
        with self.assertRaises(ValidationError):
            candidate.rank = 2  # type: ignore[misc]

    def test_canonical_id_parser_handles_underscored_video_id(self) -> None:
        for frame_id, video_id, shot_id, position_code in (
            ("L21_V001_00000_015", "L21_V001", 0, 15),
            ("L26_V498_00123_100", "L26_V498", 123, 100),
        ):
            parsed = parse_canonical_frame_id(frame_id)
            self.assertEqual(parsed.video_id, video_id)
            self.assertEqual(parsed.shot_id, shot_id)
            self.assertEqual(parsed.position_code, position_code)
        validate_canonical_frame_id(
            "L21_V001_00123_085",
            video_id="L21_V001",
            shot_id=123,
            position_code=85,
        )
        for invalid in (
            "shot_00012_pos_105",
            "L21_V001_001",
            "L21_V001_00012_105",
            " L21_V001_00000_015",
            "L21_V001_00000_015 ",
        ):
            with self.assertRaises(Exception):
                parse_canonical_frame_id(invalid)
        with self.assertRaises(Exception):
            validate_canonical_frame_id("L21_V001_00000_015", video_id="L21_V002")
        with self.assertRaises(Exception):
            validate_canonical_frame_id("L21_V001_00000_015", shot_id=2)

    def test_shared_frame_models_expose_self_indexed_identity(self) -> None:
        for model in (
            FrameSearchHit,
            FrameMetadata,
            FrameCandidate,
            FusedFrameCandidate,
            OrderedVisualFrame,
            TRAKEFrameMatch,
            ImageEvidence,
        ):
            self.assertIn("shot_id", model.model_fields)
        self.assertEqual(
            set(FrameMetadata.model_fields),
            {
                "frame_id",
                "video_id",
                "shot_id",
                "timestamp_sec",
                "source_frame_idx",
                "image_rel_path",
            },
        )
        self.assertEqual(
            set(FrameSearchHit.model_fields),
            {"frame_id", "video_id", "shot_id", "raw_score"},
        )

    def test_self_indexed_metadata_rejects_semantic_mismatches(self) -> None:
        base = build_self_indexed_frame_metadata()[0].model_dump()
        for field, value in (
            ("frame_id", "L21_V001_00001_015"),
            ("video_id", "L21_V002"),
            ("shot_id", 2),
            ("source_frame_idx", -1),
        ):
            with self.assertRaises(ValidationError):
                FrameMetadata.model_validate({**base, field: value})
        for invalid_path in (
            "C:/data/keyframe.webp",
            "/data/keyframe.webp",
            "../keyframe.webp",
            "keyframes\\V1\\frame.webp",
            "https://example.test/frame.webp",
        ):
            with self.assertRaises(ValidationError):
                FrameMetadata.model_validate({**base, "image_rel_path": invalid_path})
        with self.assertRaises(ValidationError):
            FrameSearchHit(
                frame_id="L21_V001_00000_015",
                video_id="L21_V002",
                shot_id=0,
                raw_score=0.5,
            )

    def test_self_indexed_base_fixture_preserves_duplicate_source_frame(self) -> None:
        frames = build_self_indexed_frame_metadata()
        self.assertEqual(
            tuple(frame.frame_id for frame in frames),
            (
                "L21_V001_00000_015",
                "L21_V001_00000_050",
                "L21_V001_00001_085",
                "L21_V002_00000_050",
            ),
        )
        self.assertEqual(frames[0].source_frame_idx, frames[1].source_frame_idx)
        self.assertNotEqual(frames[0].frame_id, frames[1].frame_id)
        self.assertEqual(
            tuple(type(frame).model_validate_json(frame.model_dump_json()) for frame in frames),
            frames,
        )

    def test_nested_result_mapping_is_immutable_and_serializable(self) -> None:
        from online.domain.candidates import CandidateDiagnostics, CandidateEvidence

        evidence = CandidateEvidence(
            branch=RetrievalBranch.VISUAL_DENSE,
            query_variant_id="q0",
            raw_score=0.8,
            normalized_score=0.8,
        )
        result = FusedFrameCandidate(
            frame_id="L21_V001_00000_015",
            video_id="L21_V001",
            shot_id=0,
            timestamp_sec=0.0,
            source_frame_idx=0,
            image_rel_path="keyframes/L21_V001/L21_V001_00000_015.jpg",
            final_score=0.8,
            branch_scores={RetrievalBranch.VISUAL_DENSE: 0.8},
            evidence=(evidence,),
            diagnostics=CandidateDiagnostics(),
        )
        with self.assertRaises(TypeError):
            result.branch_scores[RetrievalBranch.VISUAL_DENSE] = 0.2  # type: ignore[index]
        with self.assertRaises(TypeError):
            dict.__setitem__(result.branch_scores, RetrievalBranch.VISUAL_DENSE, 0.2)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            result.branch_scores._data = {}  # type: ignore[attr-defined]
        self.assertIn('"branch_scores"', result.model_dump_json())

    def test_strict_strings_canonical_ids_and_numeric_scores_reject_ambiguous_input(self) -> None:
        with self.assertRaises(Exception):
            parse_canonical_frame_id(" L21_V001_00000_015")
        with self.assertRaises(Exception):
            parse_canonical_frame_id("L21_V001_00000_015 ")
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "raw_score": True})
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "rank": True})
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "shot_id": True})
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "video_id": " L21_V001"})
        with self.assertRaises(ValidationError):
            ObjectConstraint(
                label="person",
                count_operator="gte",
                count=True,
            )

    def test_all_shared_enum_values_are_stable(self) -> None:
        self.assertEqual(
            tuple(mode.value for mode in QueryMode),
            ("kis_text", "kis_video", "trake", "vqa"),
        )
        self.assertEqual(
            tuple(branch.value for branch in RetrievalBranch),
            (
                "visual_dense",
                "ocr_dense",
                "ocr_bm25",
                "asr_dense",
                "asr_bm25",
                "summary_dense",
                "summary_bm25",
            ),
        )

    def test_non_success_statuses_require_reason_and_no_candidates(self) -> None:
        with self.assertRaises(ValidationError):
            BranchResult[FrameCandidate](
                branch=RetrievalBranch.VISUAL_DENSE,
                candidate_level=CandidateLevel.FRAME,
                query_variant_id="q0",
                candidates=(),
                requested_top_k=10,
                latency_ms=1,
                status=BranchStatus.DEGRADED,
            )
        candidate = FrameCandidate.model_validate(self.fixture["frame"])
        with self.assertRaises(ValidationError):
            BranchResult[FrameCandidate](
                branch=RetrievalBranch.VISUAL_DENSE,
                candidate_level=CandidateLevel.FRAME,
                query_variant_id="q0",
                candidates=(candidate,),
                requested_top_k=10,
                latency_ms=1,
                status=BranchStatus.DISABLED,
                warnings=("disabled by config",),
            )

    def test_branch_result_rejects_wrong_candidate_level(self) -> None:
        candidate = ASRIntervalCandidate.model_validate(self.fixture["asr_interval"])
        with self.assertRaises(ValidationError):
            BranchResult[ASRIntervalCandidate](
                branch=RetrievalBranch.ASR_BM25,
                candidate_level=CandidateLevel.FRAME,
                query_variant_id="q0",
                candidates=(candidate,),
                requested_top_k=10,
                latency_ms=1,
                status=BranchStatus.SUCCESS,
            )

    def test_failed_branch_requires_warning_but_success_may_be_empty(self) -> None:
        success = BranchResult[FrameCandidate](
            branch=RetrievalBranch.VISUAL_DENSE,
            candidate_level=CandidateLevel.FRAME,
            query_variant_id="q0",
            candidates=(),
            requested_top_k=10,
            latency_ms=1,
            status=BranchStatus.SUCCESS,
        )
        self.assertEqual(success.returned_count, 0)
        with self.assertRaises(ValidationError):
            BranchResult[FrameCandidate](
                branch=RetrievalBranch.VISUAL_DENSE,
                candidate_level=CandidateLevel.FRAME,
                query_variant_id="q0",
                candidates=(),
                requested_top_k=10,
                latency_ms=1,
                status=BranchStatus.FAILED,
            )

    def test_branch_result_rejects_provenance_mismatch_and_whitespace_warning(self) -> None:
        candidate = FrameCandidate.model_validate(self.fixture["frame"])
        base = {
            "candidate_level": CandidateLevel.FRAME,
            "candidates": (candidate,),
            "requested_top_k": 10,
            "latency_ms": 1,
            "status": BranchStatus.SUCCESS,
        }
        with self.assertRaises(ValidationError):
            BranchResult[FrameCandidate](
                branch=RetrievalBranch.OCR_DENSE,
                query_variant_id="q0",
                **base,
            )
        with self.assertRaises(ValidationError):
            BranchResult[FrameCandidate](
                branch=RetrievalBranch.VISUAL_DENSE,
                query_variant_id="q1",
                **base,
            )
        with self.assertRaises(ValidationError):
            BranchResult[FrameCandidate](
                branch=RetrievalBranch.VISUAL_DENSE,
                candidate_level=CandidateLevel.FRAME,
                query_variant_id="q0",
                candidates=(),
                requested_top_k=10,
                latency_ms=1,
                status=BranchStatus.FAILED,
                warnings=("   ",),
            )

        degraded = BranchResult[FrameCandidate](
            branch=RetrievalBranch.VISUAL_DENSE,
            candidate_level=CandidateLevel.FRAME,
            query_variant_id="q0",
            candidates=(candidate,),
            requested_top_k=10,
            latency_ms=1,
            status=BranchStatus.DEGRADED,
            warnings=("partial backend result",),
        )
        self.assertEqual(degraded.candidates, (candidate,))

    def test_typed_branch_result_rejects_wrong_candidate_after_json_input(self) -> None:
        payload = {
            "branch": "asr_bm25",
            "candidate_level": "frame",
            "query_variant_id": "q0",
            "candidates": [self.fixture["asr_interval"]],
            "requested_top_k": 10,
            "latency_ms": 1,
            "status": "success",
            "warnings": [],
        }
        with self.assertRaises(ValidationError):
            BranchResult[FrameCandidate].model_validate(payload)


if __name__ == "__main__":
    unittest.main()
