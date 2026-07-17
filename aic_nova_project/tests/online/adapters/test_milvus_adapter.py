from __future__ import annotations

import math
import threading
import unittest

from online.adapters.milvus import MilvusSearchAdapter
from online.config import MilvusResourceConfig
from online.domain.errors import (
    ContractMismatchError,
    DimensionMismatchError,
    InvalidQueryError,
    ResourceUnavailableError,
)


class FakeBackend:
    def __init__(self) -> None:
        self.connected = False
        self.raise_error: Exception | None = None
        self.descriptions = {
            name: {
                "fields": {"embedding": "FLOAT_VECTOR"},
                "dimension": 2,
                "metric_type": "IP",
                "index_type": "HNSW",
            }
            for name in ("visual_features", "ocr_features", "asr_features", "summary_features")
        }
        self.hits = {}
        self.last_search = None

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def collection_exists(self, name: str) -> bool:
        return name in self.descriptions

    def describe_collection(self, name: str):
        return self.descriptions[name]

    def search(self, name, vector, output_fields, top_k, search_params, timeout_sec):
        if self.raise_error:
            raise self.raise_error
        self.last_search = (name, vector, output_fields, top_k, search_params, timeout_sec)
        return self.hits.get(name, ())

    def sample_records(self, name, output_fields, limit):
        return ()


class MilvusAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeBackend()
        self.adapter = MilvusSearchAdapter(MilvusResourceConfig(), backend=self.backend)
        self.adapter.connect()
        self.vector = (1 / math.sqrt(2), 1 / math.sqrt(2))

    def test_maps_all_collection_levels_without_pk(self) -> None:
        self.backend.hits = {
            "visual_features": ({"entity": {"frame_id": "F1", "video_id": "V1", "shot_id": 2}, "distance": 0.9, "pk": 99},),
            "ocr_features": ({"entity": {"frame_id": "F1", "video_id": "V1"}, "distance": 0.8},),
            "asr_features": ({"entity": {"video_id": "V1", "interval_id": "i1", "start_time_sec": 1, "end_time_sec": 2}, "distance": 0.7},),
            "summary_features": ({"entity": {"video_id": "V1"}, "distance": 0.6},),
        }
        self.assertEqual(self.adapter.search_visual(self.vector, 5)[0].frame_id, "F1")
        self.assertIsNone(self.adapter.search_ocr(self.vector, 5)[0].shot_id)
        self.assertEqual(self.adapter.search_asr(self.vector, 5)[0].interval_id, "i1")
        self.assertEqual(self.adapter.search_summary(self.vector, 5)[0].video_id, "V1")
        self.assertEqual(self.backend.last_search[4], {"metric_type": "IP", "params": {"ef": 128}})

    def test_dimension_finite_and_norm_validation(self) -> None:
        with self.assertRaises(DimensionMismatchError):
            self.adapter.search_visual((1.0,), 1)
        with self.assertRaises(InvalidQueryError):
            self.adapter.search_visual((float("nan"), 0.0), 1)
        with self.assertRaises(InvalidQueryError):
            self.adapter.search_visual((1.0, 1.0), 1)

    def test_empty_result_differs_from_backend_failure(self) -> None:
        self.assertEqual(self.adapter.search_visual(self.vector, 1), ())
        self.backend.raise_error = ConnectionError("down")
        with self.assertRaises(ResourceUnavailableError):
            self.adapter.search_visual(self.vector, 1)

    def test_missing_field_and_timeout_are_explicit(self) -> None:
        self.backend.hits["visual_features"] = ({"entity": {"video_id": "V1", "shot_id": 1}, "distance": 0.5},)
        with self.assertRaises(ContractMismatchError):
            self.adapter.search_visual(self.vector, 1)
        self.backend.raise_error = TimeoutError("slow")
        with self.assertRaises(Exception) as raised:
            self.adapter.search_ocr(self.vector, 1)
        self.assertEqual(raised.exception.code.value, "BRANCH_TIMEOUT")

    def test_malformed_score_and_sample_response_are_contract_errors(self) -> None:
        self.backend.hits["ocr_features"] = (
            {"entity": {"frame_id": "F1", "video_id": "V1"}, "distance": float("nan")},
        )
        with self.assertRaises(ContractMismatchError):
            self.adapter.search_ocr(self.vector, 1)
        self.backend.sample_records = lambda name, output_fields, limit: "bad"  # type: ignore[method-assign]
        with self.assertRaises(ContractMismatchError):
            self.adapter.sample_records("visual_features", ("embedding",), 1)

    def test_close_is_rejected_while_a_read_is_active(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original = self.backend.search

        def blocked(*args, **kwargs):
            entered.set()
            release.wait(timeout=1)
            return original(*args, **kwargs)

        self.backend.search = blocked
        thread = threading.Thread(
            target=lambda: self.adapter.search_visual(self.vector, 1)
        )
        thread.start()
        self.assertTrue(entered.wait(timeout=1))
        with self.assertRaises(ResourceUnavailableError):
            self.adapter.close()
        release.set()
        thread.join(timeout=2)
        self.adapter.close()


if __name__ == "__main__":
    unittest.main()
