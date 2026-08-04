from __future__ import annotations

import unittest

from online.domain.candidates import ASRIntervalCandidate
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.domain.errors import ContractMismatchError, InvalidQueryError, ResourceUnavailableError
from online.domain.query import TextQueryVariant
from online.ports.records import VideoSearchHit
from online.retrieval.branches import ASRLexicalBranch, ASRSemanticBranch
from online.retrieval.query_builder import KISQueryBuilder
from online.testing import (
    FakeElasticsearchSearchPort,
    FakeMilvusSearchPort,
    FakeTextEncoder,
    build_integration_fixture,
)


class StepClock:
    def __init__(self, step_sec: float = 0.01) -> None:
        self.value = 20.0
        self.step_sec = step_sec

    def __call__(self) -> float:
        value = self.value
        self.value += self.step_sec
        return value


class BrokenElasticsearchPort(FakeElasticsearchSearchPort):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def search_asr(self, query, top_k, *, fuzzy=None):
        raise self.error


class BrokenASRMilvusPort(FakeMilvusSearchPort):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def search_asr(self, vector, top_k):
        raise self.error


class InvalidASRPort(FakeMilvusSearchPort):
    def search_asr(self, vector, top_k):
        return (VideoSearchHit(video_id="V001", raw_score=0.5),)


class ASRBranchTests(unittest.TestCase):
    def test_lexical_uses_q0_and_preserves_interval_text_score_and_provenance(self) -> None:
        fixture = build_integration_fixture()
        elasticsearch = FakeElasticsearchSearchPort(asr=fixture.asr_hits)
        branch = ASRLexicalBranch(
            elasticsearch=elasticsearch,
            fuzzy=False,
            source_resource="configured_asr_index",
            clock=StepClock(),
        )
        query = KISQueryBuilder().build(
            "spoken bicycle",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("speech about a bike", "bicycle mentioned aloud"),
        )

        results = branch.retrieve(query, top_k=2)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.branch, RetrievalBranch.ASR_BM25)
        self.assertEqual(result.candidate_level, CandidateLevel.ASR_INTERVAL)
        self.assertEqual(result.query_variant_id, "q0")
        self.assertEqual(result.status, BranchStatus.SUCCESS)
        self.assertEqual(result.requested_top_k, 2)
        self.assertAlmostEqual(result.latency_ms, 10.0)
        self.assertEqual(elasticsearch.asr_calls, [("spoken bicycle", 2, False)])

        candidate = result.candidates[0]
        self.assertIsInstance(candidate, ASRIntervalCandidate)
        self.assertEqual(candidate.video_id, fixture.asr_hits[0].video_id)
        self.assertEqual(candidate.interval_id, fixture.asr_hits[0].interval_id)
        self.assertEqual(candidate.start_time_sec, fixture.asr_hits[0].start_time_sec)
        self.assertEqual(candidate.end_time_sec, fixture.asr_hits[0].end_time_sec)
        self.assertEqual(candidate.text, fixture.asr_hits[0].text)
        self.assertEqual(candidate.rank, 1)
        self.assertEqual(candidate.raw_score, fixture.asr_hits[0].raw_score)
        self.assertIsNone(candidate.normalized_score)
        self.assertEqual(candidate.provenance.backend, "elasticsearch")
        self.assertEqual(candidate.provenance.source_resource, "configured_asr_index")
        self.assertEqual(candidate.provenance.query_text, "spoken bicycle")
        self.assertFalse(hasattr(candidate, "frame_id"))

    def test_semantic_runs_q0_q1_q2_independently_without_frame_mapping(self) -> None:
        fixture = build_integration_fixture()
        encoder = FakeTextEncoder(dimension=5)
        milvus = FakeMilvusSearchPort(asr=fixture.asr_hits)
        branch = ASRSemanticBranch(
            encoder=encoder,
            milvus=milvus,
            clock=StepClock(),
        )
        query = KISQueryBuilder().build(
            "original ASR query",
            mode=QueryMode.KIS_VIDEO,
            paraphrases=("ASR paraphrase one", "ASR paraphrase two"),
        )

        results = branch.retrieve(query, top_k=3)

        self.assertEqual(tuple(result.query_variant_id for result in results), ("q0", "q1", "q2"))
        self.assertEqual(
            encoder.calls,
            [("original ASR query",), ("ASR paraphrase one",), ("ASR paraphrase two",)],
        )
        self.assertEqual(len(milvus.asr_calls), 3)
        self.assertEqual(len({call[0] for call in milvus.asr_calls}), 3)
        self.assertTrue(all(call[1] == 3 for call in milvus.asr_calls))
        self.assertTrue(all(result.branch is RetrievalBranch.ASR_DENSE for result in results))
        self.assertTrue(all(result.returned_count == 3 for result in results))
        self.assertTrue(
            all(
                isinstance(candidate, ASRIntervalCandidate) and not hasattr(candidate, "frame_id")
                for result in results
                for candidate in result.candidates
            )
        )
        self.assertEqual(results[0].candidates[2].interval_id, "no_overlap")
        self.assertEqual(results[0].candidates[2].rank, 3)
        self.assertEqual(results[0].candidates[0].provenance.backend, "milvus")
        self.assertEqual(results[0].candidates[0].provenance.source_resource, "asr_features")

    def test_empty_lexical_and_semantic_results_are_success(self) -> None:
        query = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        lexical = ASRLexicalBranch(
            elasticsearch=FakeElasticsearchSearchPort(),
            clock=StepClock(),
        )
        semantic = ASRSemanticBranch(
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
        lexical = ASRLexicalBranch(elasticsearch=elasticsearch)
        semantic = ASRSemanticBranch(encoder=encoder, milvus=milvus)

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
        self.assertEqual(elasticsearch.asr_calls, [])
        self.assertEqual(milvus.asr_calls, [])
        self.assertEqual(encoder.calls, [])

    def test_asr_backend_and_port_contract_failures_surface(self) -> None:
        query = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        elasticsearch_error = ResourceUnavailableError("Elasticsearch unavailable")
        lexical = ASRLexicalBranch(
            elasticsearch=BrokenElasticsearchPort(elasticsearch_error)
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            lexical.retrieve(query, top_k=1)
        self.assertIs(raised.exception, elasticsearch_error)

        milvus_error = ResourceUnavailableError("Milvus unavailable")
        semantic = ASRSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=BrokenASRMilvusPort(milvus_error),
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            semantic.retrieve(query, top_k=1)
        self.assertIs(raised.exception, milvus_error)

        malformed = ASRSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=InvalidASRPort(),
        )
        with self.assertRaises(ContractMismatchError):
            malformed.retrieve(query, top_k=1)

    def test_invalid_fuzzy_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ASRLexicalBranch(
                elasticsearch=FakeElasticsearchSearchPort(),
                fuzzy="AUTO",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
