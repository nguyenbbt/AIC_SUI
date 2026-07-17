from __future__ import annotations

import unittest

from online.adapters.elasticsearch import ElasticsearchSearchAdapter
from online.config import ElasticsearchResourceConfig
from online.domain.errors import ContractMismatchError, InvalidQueryError, ResourceUnavailableError


class FakeIndices:
    def exists(self, *, index):
        return True

    def get_mapping(self, *, index):
        return {index: {"mappings": {"properties": {"summary": {"type": "text", "analyzer": "vietnamese_analyzer"}}}}}


class FakeClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.response = {"hits": {"hits": []}}
        self.last_call = None
        self.error = None

    def search(self, **kwargs):
        if self.error:
            raise self.error
        self.last_call = kwargs
        return self.response

    def ping(self):
        return True


class ElasticsearchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.adapter = ElasticsearchSearchAdapter(
            ElasticsearchResourceConfig(), client=self.client
        )

    def test_exact_query_body_and_ocr_mapping(self) -> None:
        self.client.response = {"hits": {"hits": [{"_score": 4.5, "_source": {"frame_id": "F1", "video_id": "V1", "shot_id": "2"}}]}}
        result = self.adapter.search_ocr("xin chào", 7, fuzzy=False)
        self.assertEqual(result[0].shot_id, 2)
        self.assertEqual(
            self.client.last_call["body"],
            {
                "size": 7,
                "_source": ["frame_id", "video_id", "shot_id"],
                "query": {"match": {"ocr_text_concat": {"query": "xin chào"}}},
            },
        )

    def test_fuzzy_body_asr_and_summary_mapping(self) -> None:
        self.client.response = {"hits": {"hits": [{"_score": 3, "_source": {"video_id": "V1", "interval_id": "i1", "start_time": 1, "end_time": 2, "cleaned_text": "text"}}]}}
        asr = self.adapter.search_asr("hello", 2, fuzzy=True)
        self.assertEqual(asr[0].start_time_sec, 1)
        self.assertEqual(self.client.last_call["body"]["query"]["match"]["cleaned_text"]["fuzziness"], "AUTO")
        self.client.response = {"hits": {"hits": [{"_score": 2, "_source": {"video_id": "V1", "summary": "sum"}}]}}
        self.assertEqual(self.adapter.search_summary("hello", 2)[0].summary, "sum")

    def test_empty_hits_differs_from_failure_and_empty_query_is_rejected(self) -> None:
        self.assertEqual(self.adapter.search_summary("hello", 1), ())
        with self.assertRaises(InvalidQueryError):
            self.adapter.search_summary(" ", 1)
        self.client.error = ConnectionError("down")
        with self.assertRaises(ResourceUnavailableError):
            self.adapter.search_summary("hello", 1)

    def test_missing_score_or_source_field_is_contract_mismatch(self) -> None:
        self.client.response = {"hits": {"hits": [{"_source": {"video_id": "V1"}}]}}
        with self.assertRaises(ContractMismatchError):
            self.adapter.search_summary("hello", 1)

    def test_find_documents_validates_limit_and_malformed_hits(self) -> None:
        with self.assertRaises(InvalidQueryError):
            self.adapter.find_documents("video_summaries", {"video_id": "V1"}, ("video_id",), limit=0)
        self.client.response = {"hits": {"hits": [{"_score": 1, "_source": None}]}}
        with self.assertRaises(ContractMismatchError):
            self.adapter.sample_documents("video_summaries", ("video_id",), 1)
        self.client.response = {"hits": {"hits": [{"_score": 1, "_source": {}}]}}
        with self.assertRaises(ContractMismatchError):
            self.adapter.search_summary("hello", 1)


if __name__ == "__main__":
    unittest.main()
