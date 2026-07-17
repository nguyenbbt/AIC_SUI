from __future__ import annotations

import asyncio
import unittest
from collections import Counter

from pydantic import ValidationError

from online.domain.candidates import BranchResult
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.domain.query import QueryBundle, TextQueryVariant
from online.retrieval.query_builder import BASELINE_KIS_BRANCHES
from online.retrieval.service import (
    MULTI_VARIANT_BRANCHES,
    RetrievalInvocationConfig,
    RetrievalService,
)
from query_understanding import parse_kis_query


_CANDIDATE_LEVELS = {
    RetrievalBranch.VISUAL_DENSE: CandidateLevel.FRAME,
    RetrievalBranch.OCR_DENSE: CandidateLevel.FRAME,
    RetrievalBranch.OCR_BM25: CandidateLevel.FRAME,
    RetrievalBranch.ASR_DENSE: CandidateLevel.ASR_INTERVAL,
    RetrievalBranch.ASR_BM25: CandidateLevel.ASR_INTERVAL,
    RetrievalBranch.SUMMARY_DENSE: CandidateLevel.VIDEO,
    RetrievalBranch.SUMMARY_BM25: CandidateLevel.VIDEO,
}


class _RecordingRunner:
    def __init__(self, branch: RetrievalBranch) -> None:
        self.branch = branch
        self.calls: list[tuple[str, str, int]] = []

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult:
        self.calls.append((variant.variant_id, variant.text, top_k))
        return BranchResult(
            branch=self.branch,
            candidate_level=_CANDIDATE_LEVELS[self.branch],
            query_variant_id=variant.variant_id,
            candidates=(),
            requested_top_k=top_k,
            latency_ms=0.0,
            status=BranchStatus.SUCCESS,
        )


class KISModeParityTests(unittest.TestCase):
    def test_textual_and_video_kis_use_the_same_retrieval_service_and_plan(self) -> None:
        common = {
            "original_query": "một người mặc áo đỏ đang đi xe đạp",
            "paraphrases": (
                "người áo đỏ đạp xe",
                "một người đang chạy xe đạp với áo màu đỏ",
            ),
        }
        textual = parse_kis_query(
            mode=QueryMode.KIS_TEXT,
            query_id="textual-kis",
            **common,
        )
        video = parse_kis_query(
            mode=QueryMode.KIS_VIDEO,
            query_id="video-kis",
            **common,
        )

        self.assertNotEqual(textual.mode, video.mode)
        self.assertEqual(textual.original_query, video.original_query)
        self.assertEqual(textual.text_variants, video.text_variants)
        self.assertEqual(textual.enabled_branches, video.enabled_branches)

        runners = {
            branch: _RecordingRunner(branch)
            for branch in BASELINE_KIS_BRANCHES
        }
        configs = {}
        for branch in BASELINE_KIS_BRANCHES:
            variants = (
                textual.text_variants
                if branch in MULTI_VARIANT_BRANCHES
                else textual.text_variants[:1]
            )
            for variant in variants:
                configs[(branch, variant.variant_id)] = RetrievalInvocationConfig(
                    top_k=5,
                    timeout_sec=1.0,
                )

        service = RetrievalService(
            branches=runners,
            invocation_configs=configs,
            max_workers=7,
        )

        async def retrieve_both_modes():
            textual_results = await service.retrieve(textual)
            calls_after_textual = {
                branch: Counter(runner.calls)
                for branch, runner in runners.items()
            }
            video_results = await service.retrieve(video)
            return textual_results, calls_after_textual, video_results

        try:
            textual_results, calls_after_textual, video_results = asyncio.run(
                retrieve_both_modes()
            )
        finally:
            service.close(wait=True)

        self.assertEqual(textual_results, video_results)
        self.assertEqual(
            tuple((result.branch, result.query_variant_id) for result in textual_results),
            tuple((result.branch, result.query_variant_id) for result in video_results),
        )
        for branch, runner in runners.items():
            after_video = Counter(runner.calls)
            self.assertEqual(
                after_video,
                calls_after_textual[branch] + calls_after_textual[branch],
            )

    def test_video_kis_bundle_rejects_machine_readable_video_input(self) -> None:
        bundle = parse_kis_query(
            "mô tả do thí sinh tự viết sau khi xem clip",
            mode=QueryMode.KIS_VIDEO,
            query_id="manual-video-description",
        )
        payload = bundle.model_dump(mode="json")
        payload["video_path"] = "organizer-clip.mp4"

        with self.assertRaises(ValidationError):
            QueryBundle.model_validate(payload)

        self.assertNotIn("video_path", QueryBundle.model_fields)
        self.assertNotIn("image", QueryBundle.model_fields)


if __name__ == "__main__":
    unittest.main()
