from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    FrameCandidate,
    VideoCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.domain.identifiers import parse_canonical_frame_id
from online.domain.errors import ResourceUnavailableError
from online.domain.query import ObjectConstraint


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

    def test_object_constraint_rejects_negative_count(self) -> None:
        with self.assertRaises(ValidationError):
            ObjectConstraint(
                label="person",
                count_operator="gte",
                count=-1,
                min_confidence=0.5,
            )

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
        parsed = parse_canonical_frame_id("camera_north_V001_00012_105")
        self.assertEqual(parsed.video_id, "camera_north_V001")
        self.assertEqual(parsed.shot_id, 12)
        self.assertEqual(parsed.position, 105)
        with self.assertRaises(Exception):
            parse_canonical_frame_id("shot_00012_pos_105")

    def test_nested_result_mapping_is_immutable_and_serializable(self) -> None:
        from online.domain.candidates import (
            CandidateDiagnostics,
            CandidateEvidence,
            FusedFrameCandidate,
        )

        evidence = CandidateEvidence(
            branch=RetrievalBranch.VISUAL_DENSE,
            query_variant_id="q0",
            raw_score=0.8,
            normalized_score=0.8,
        )
        result = FusedFrameCandidate(
            frame_id="V001_00000_015",
            video_id="V001",
            shot_id=0,
            timestamp_sec=1.5,
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
            parse_canonical_frame_id(" V001_00000_015")
        with self.assertRaises(Exception):
            parse_canonical_frame_id("V001_00000_015 ")
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "raw_score": True})
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "rank": True})
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "shot_id": True})
        with self.assertRaises(ValidationError):
            FrameCandidate.model_validate({**self.fixture["frame"], "video_id": " V001"})
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
