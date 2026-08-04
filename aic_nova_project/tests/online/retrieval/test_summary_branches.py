from __future__ import annotations

import unittest

from online.domain.candidates import VideoCandidate
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.domain.errors import ContractMismatchError, InvalidQueryError, ResourceUnavailableError
from online.domain.query import TextQueryVariant
from online.ports.records import ASRSearchHit, VideoSearchHit
from online.retrieval.branches import SummaryLexicalBranch, SummarySemanticBranch
from online.retrieval.query_builder import KISQueryBuilder
from online.testing import FakeElasticsearchSearchPort, FakeMilvusSearchPort, FakeTextEncoder


class StepClock:
    def __init__(self, step_sec: float = 0.01) -> None:
        self.value = 30.0
        self.step_sec = step_sec

    def __call__(self) -> float:
        value = self.value
        self.value += self.step_sec
        return value


class BrokenElasticsearchPort(FakeElasticsearchSearchPort):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def search_summary(self, query, top_k, *, fuzzy=None):
        raise self.error


class BrokenSummaryMilvusPort(FakeMilvusSearchPort):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def search_summary(self, vector, top_k):
        raise self.error


class InvalidSummaryPort(FakeMilvusSearchPort):
    def search_summary(self, vector, top_k):
        return (ASRSearchHit(video_id="V001", interval_id="i1", start_time_sec=0, end_time_sec=1, raw_score=0.5),)


class SummaryBranchTests(unittest.TestCase):
    def test_lexical_uses_q0_and_preserves_summary_score_and_video_provenance(self) -> None:
        hits = (
            VideoSearchHit(video_id="V001", raw_score=8.2, summary="A person rides a bicycle."),
            VideoSearchHit(video_id="V002", raw_score=6.1, summary="A red car stops."),
        )
        elasticsearch = FakeElasticsearchSearchPort(summary=hits)
        branch = SummaryLexicalBranch(
            elasticsearch=elasticsearch,
            fuzzy=True,
            source_resource="configured_summary_index",
            clock=StepClock(),
        )
        query = KISQueryBuilder().build(
            "video about a bicycle",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("bicycle clip", "someone cycling"),
        )

        results = branch.retrieve(query, top_k=2)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.branch, RetrievalBranch.SUMMARY_BM25)
        self.assertEqual(result.candidate_level, CandidateLevel.VIDEO)
        self.assertEqual(result.query_variant_id, "q0")
        self.assertEqual(result.status, BranchStatus.SUCCESS)
        self.assertEqual(result.requested_top_k, 2)
        self.assertAlmostEqual(result.latency_ms, 10.0)
        self.assertEqual(
            elasticsearch.summary_calls,
            [("video about a bicycle", 2, True)],
        )

        candidate = result.candidates[0]
        self.assertIsInstance(candidate, VideoCandidate)
        self.assertEqual(candidate.video_id, "V001")
        self.assertEqual(candidate.summary, "A person rides a bicycle.")
        self.assertEqual(candidate.rank, 1)
        self.assertEqual(candidate.raw_score, 8.2)
        self.assertIsNone(candidate.normalized_score)
        self.assertEqual(candidate.provenance.backend, "elasticsearch")
        self.assertEqual(candidate.provenance.source_resource, "configured_summary_index")
        self.assertEqual(candidate.provenance.query_text, "video about a bicycle")
        self.assertFalse(hasattr(candidate, "frame_id"))
        self.assertFalse(hasattr(candidate, "final_score"))

    def test_semantic_runs_all_variants_and_does_not_deduplicate_videos(self) -> None:
        hits = (
            VideoSearchHit(video_id="V001", raw_score=0.91),
            VideoSearchHit(video_id="V001", raw_score=0.85),
            VideoSearchHit(video_id="V002", raw_score=0.75),
        )
        encoder = FakeTextEncoder(dimension=5)
        milvus = FakeMilvusSearchPort(summary=hits)
        branch = SummarySemanticBranch(
            encoder=encoder,
            milvus=milvus,
            clock=StepClock(),
        )
        query = KISQueryBuilder().build(
            "original summary query",
            mode=QueryMode.KIS_VIDEO,
            paraphrases=("summary paraphrase one", "summary paraphrase two"),
        )

        results = branch.retrieve(query, top_k=3)

        self.assertEqual(tuple(result.query_variant_id for result in results), ("q0", "q1", "q2"))
        self.assertEqual(
            encoder.calls,
            [
                ("original summary query",),
                ("summary paraphrase one",),
                ("summary paraphrase two",),
            ],
        )
        self.assertEqual(len(milvus.summary_calls), 3)
        self.assertEqual(len({call[0] for call in milvus.summary_calls}), 3)
        self.assertTrue(all(call[1] == 3 for call in milvus.summary_calls))
        self.assertTrue(all(result.branch is RetrievalBranch.SUMMARY_DENSE for result in results))
        self.assertTrue(all(result.returned_count == 3 for result in results))
        self.assertEqual(
            tuple(candidate.video_id for candidate in results[0].candidates),
            ("V001", "V001", "V002"),
        )
        self.assertEqual(tuple(candidate.rank for candidate in results[0].candidates), (1, 2, 3))
        self.assertEqual(results[0].candidates[0].provenance.backend, "milvus")
        self.assertEqual(results[0].candidates[0].provenance.source_resource, "summary_features")
        self.assertTrue(all(not hasattr(candidate, "frame_id") for candidate in results[0].candidates))

    def test_empty_lexical_and_semantic_results_are_success(self) -> None:
        query = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        lexical = SummaryLexicalBranch(
            elasticsearch=FakeElasticsearchSearchPort(),
            clock=StepClock(),
        )
        semantic = SummarySemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=FakeMilvusSearchPort(),
            clock=StepClock(),
        )

        lexical_result = lexical.retrieve(query, top_k=4)[0]
        semantic_result = semantic.retrieve(query, top_k=4)[0]

        self.assertEqual(lexical_result.status, BranchStatus.SUCCESS)
        self.assertEqual(semantic_result.status, BranchStatus.SUCCESS)
        self.assertEqual(lexical_result.candidates, ())
        self.assertEqual(semantic_result.candidates, ())
        self.assertEqual(lexical_result.warnings, ())
        self.assertEqual(semantic_result.warnings, ())

    def test_lexical_rejects_q1_and_disabled_branches_do_no_work(self) -> None:
        elasticsearch = FakeElasticsearchSearchPort()
        milvus = FakeMilvusSearchPort()
        encoder = FakeTextEncoder()
        lexical = SummaryLexicalBranch(elasticsearch=elasticsearch)
        semantic = SummarySemanticBranch(encoder=encoder, milvus=milvus)

        with self.assertRaises(InvalidQueryError):
            lexical.retrieve_variant(
                TextQueryVariant(variant_id="q1", text="paraphrase"),
                top_k=1,
            )
        disabled = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.VISUAL_DENSE,),
        )
        with self.assertRaises(InvalidQueryError):
            lexical.retrieve(disabled, top_k=1)
        with self.assertRaises(InvalidQueryError):
            semantic.retrieve(disabled, top_k=1)
        self.assertEqual(elasticsearch.summary_calls, [])
        self.assertEqual(milvus.summary_calls, [])
        self.assertEqual(encoder.calls, [])

    def test_summary_backend_and_port_contract_failures_surface(self) -> None:
        query = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        elasticsearch_error = ResourceUnavailableError("Elasticsearch unavailable")
        lexical = SummaryLexicalBranch(
            elasticsearch=BrokenElasticsearchPort(elasticsearch_error)
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            lexical.retrieve(query, top_k=1)
        self.assertIs(raised.exception, elasticsearch_error)

        milvus_error = ResourceUnavailableError("Milvus unavailable")
        semantic = SummarySemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=BrokenSummaryMilvusPort(milvus_error),
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            semantic.retrieve(query, top_k=1)
        self.assertIs(raised.exception, milvus_error)

        malformed = SummarySemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=InvalidSummaryPort(),
        )
        with self.assertRaises(ContractMismatchError):
            malformed.retrieve(query, top_k=1)

    def test_invalid_fuzzy_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SummaryLexicalBranch(
                elasticsearch=FakeElasticsearchSearchPort(),
                fuzzy="AUTO",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
