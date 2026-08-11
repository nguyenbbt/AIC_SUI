from __future__ import annotations

import unittest

from pydantic import ValidationError

from online.domain.enums import QueryMode, RetrievalBranch
from online.domain.errors import InvalidQueryError
from online.domain.query import QueryBundle, TextQueryVariant
from online.retrieval.query_builder import BASELINE_KIS_BRANCHES, KISQueryBuilder
from query_understanding import parse_kis_query


class KISQueryBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = KISQueryBuilder()

    def test_builds_q0_q1_and_preserves_structured_constraints(self) -> None:
        bundle = self.builder.build(
            "  một người đi xe đạp  ",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("người đang đạp xe",),
            object_constraints=(
                {
                    "label": "person",
                    "count_operator": "gte",
                    "count": 1,
                    "min_confidence": 0.5,
                    "filter_mode": "soft",
                },
            ),
            query_id="query-001",
        )

        self.assertEqual(bundle.query_id, "query-001")
        self.assertEqual(bundle.original_query, "một người đi xe đạp")
        self.assertEqual(
            tuple(variant.variant_id for variant in bundle.text_variants),
            ("q0", "q1"),
        )
        self.assertEqual(bundle.text_variants[0].text, bundle.original_query)
        self.assertEqual(bundle.enabled_branches, BASELINE_KIS_BRANCHES)
        self.assertEqual(bundle.object_constraints[0].label, "person")
        self.assertIsNone(bundle.text_variants[0].weight_hint)

    def test_textual_and_video_kis_share_the_same_query_contract(self) -> None:
        common = {
            "original_query": "một chiếc xe màu đỏ",
            "paraphrases": ("xe hơi màu đỏ",),
            "query_id": "same-request",
        }
        textual = self.builder.build(mode=QueryMode.KIS_TEXT, **common)
        video = self.builder.build(mode=QueryMode.KIS_VIDEO, **common)

        self.assertEqual(textual.original_query, video.original_query)
        self.assertEqual(textual.text_variants, video.text_variants)
        self.assertEqual(textual.enabled_branches, video.enabled_branches)
        self.assertEqual(textual.object_constraints, video.object_constraints)
        self.assertNotEqual(textual.mode, video.mode)
        self.assertNotIn("image", textual.model_fields_set)
        self.assertNotIn("image", video.model_fields_set)

    def test_rejects_empty_query_invalid_mode_and_too_many_paraphrases(self) -> None:
        with self.assertRaises(InvalidQueryError):
            self.builder.build(" ", mode=QueryMode.KIS_TEXT)
        with self.assertRaises(InvalidQueryError):
            self.builder.build("query", mode=QueryMode.KIS_TEXT, query_id="")
        with self.assertRaises(InvalidQueryError):
            self.builder.build("query", mode=QueryMode.TRAKE)
        with self.assertRaises(InvalidQueryError):
            self.builder.build("query", mode="unknown")
        with self.assertRaises(InvalidQueryError):
            self.builder.build("query", mode=QueryMode.KIS_TEXT, paraphrases=("a", "b"))

    def test_rejects_blank_paraphrase_duplicate_branches_and_empty_branch_set(self) -> None:
        with self.assertRaises(InvalidQueryError):
            self.builder.build("query", mode=QueryMode.KIS_TEXT, paraphrases=(" ",))
        with self.assertRaises(InvalidQueryError):
            self.builder.build(
                "query",
                mode=QueryMode.KIS_TEXT,
                enabled_branches=(RetrievalBranch.VISUAL_DENSE,) * 2,
            )
        with self.assertRaises(InvalidQueryError):
            self.builder.build("query", mode=QueryMode.KIS_TEXT, enabled_branches=())

    def test_query_bundle_rejects_non_contiguous_or_duplicate_variant_ids(self) -> None:
        with self.assertRaises(ValidationError):
            QueryBundle(
                query_id="query-001",
                mode=QueryMode.KIS_TEXT,
                original_query="query",
                text_variants=(
                    TextQueryVariant(variant_id="q0", text="query"),
                    TextQueryVariant(variant_id="q0", text="paraphrase"),
                ),
                enabled_branches=(RetrievalBranch.VISUAL_DENSE,),
            )

    def test_query_bundle_round_trip_and_parser_facade(self) -> None:
        bundle = parse_kis_query(
            "query",
            mode="kis_video",
            paraphrases=("paraphrase",),
            enabled_branches=("visual_dense", "ocr_bm25"),
            query_id="query-002",
        )
        restored = QueryBundle.model_validate_json(bundle.model_dump_json())

        self.assertEqual(restored, bundle)
        self.assertEqual(restored.mode, QueryMode.KIS_VIDEO)
        self.assertEqual(
            restored.enabled_branches,
            (RetrievalBranch.VISUAL_DENSE, RetrievalBranch.OCR_BM25),
        )


if __name__ == "__main__":
    unittest.main()
