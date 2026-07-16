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
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
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

    def test_models_are_frozen(self) -> None:
        candidate = FrameCandidate.model_validate(self.fixture["frame"])
        with self.assertRaises(ValidationError):
            candidate.rank = 2  # type: ignore[misc]

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


if __name__ == "__main__":
    unittest.main()
