from __future__ import annotations

import unittest

from online.adapters.contract_validator import (
    CheckStatus,
    OfflineContractValidator,
    ValidationStatus,
)
from online.config import OnlineDataConfig
from online.ports.records import FrameMetadata


class FakeMilvusInspector:
    def __init__(self) -> None:
        self.records = {
            "visual_features": [{"frame_id": "V1_00000_050", "video_id": "V1", "shot_id": 0, "embedding": [1.0, 0.0]}],
            "ocr_features": [{"frame_id": "V1_00000_050", "video_id": "V1", "embedding": [1.0, 0.0]}],
            "asr_features": [{"video_id": "V1", "interval_id": "i1", "start_time_sec": 0.0, "end_time_sec": 2.0, "embedding": [1.0, 0.0]}],
            "summary_features": [{"video_id": "V1", "embedding": [1.0, 0.0]}],
        }
        self.fields = {
            "visual_features": {"frame_id": "VARCHAR", "video_id": "VARCHAR", "shot_id": "INT64", "embedding": "FLOAT_VECTOR"},
            "ocr_features": {"frame_id": "VARCHAR", "video_id": "VARCHAR", "embedding": "FLOAT_VECTOR"},
            "asr_features": {"video_id": "VARCHAR", "interval_id": "VARCHAR", "start_time_sec": "FLOAT", "end_time_sec": "FLOAT", "embedding": "FLOAT_VECTOR"},
            "summary_features": {"video_id": "VARCHAR", "embedding": "FLOAT_VECTOR"},
        }

    def collection_exists(self, name):
        return name in self.records

    def describe_collection(self, name):
        return {"fields": self.fields[name], "dimension": 2, "metric_type": "IP", "index_type": "HNSW"}

    def sample_records(self, name, output_fields, limit):
        return tuple(
            {field: record[field] for field in output_fields}
            for record in self.records[name][:limit]
        )


class FakeElasticsearchInspector:
    TYPES = {
        "ocr_texts": {"frame_id": "keyword", "video_id": "keyword", "shot_id": "keyword", "ocr_text_concat": "text"},
        "asr_transcripts": {"video_id": "keyword", "interval_id": "keyword", "start_time": "float", "end_time": "float", "cleaned_text": "text"},
        "video_summaries": {"video_id": "keyword", "summary": "text"},
    }

    def __init__(self) -> None:
        self.documents = {
            "ocr_texts": [{"frame_id": "V1_00000_050", "video_id": "V1", "shot_id": "0", "ocr_text_concat": "text"}],
            "asr_transcripts": [{"video_id": "V1", "interval_id": "i1", "start_time": 0.0, "end_time": 2.0, "cleaned_text": "speech"}],
            "video_summaries": [{"video_id": "V1", "summary": "summary"}],
        }

    def index_exists(self, name):
        return name in self.documents

    def get_mapping(self, name):
        return {
            "properties": {
                field: {
                    "type": kind,
                    **({"analyzer": "vietnamese_analyzer"} if kind == "text" else {}),
                }
                for field, kind in self.TYPES[name].items()
            }
        }

    def sample_documents(self, name, source_fields, limit):
        return tuple(
            {field: document[field] for field in source_fields}
            for document in self.documents[name][:limit]
        )

    def find_documents(self, name, filters, source_fields, limit=2):
        return tuple(
            {field: document[field] for field in source_fields}
            for document in self.documents[name]
            if all(document.get(field) == value for field, value in filters.items())
        )[:limit]

    def has_icu_plugin(self):
        return True


class FakeSQLiteInspector:
    COLUMNS = {
        "metadata": {"frame_id": "TEXT", "video_id": "TEXT", "shot_id": "INTEGER", "timestamp": "REAL"},
        "objects": {"id": "INTEGER", "frame_id": "TEXT", "label": "TEXT", "confidence": "REAL", "x_min": "REAL", "y_min": "REAL", "x_max": "REAL", "y_max": "REAL", "model_source": "TEXT"},
    }

    def __init__(self) -> None:
        self.metadata = {
            "V1_00000_050": FrameMetadata(frame_id="V1_00000_050", video_id="V1", shot_id=0, timestamp_sec=1.0)
        }
        self.rows = {
            "metadata": [{"frame_id": "V1_00000_050", "video_id": "V1", "shot_id": 0, "timestamp": 1.0}],
            "objects": [{"id": 1, "frame_id": "V1_00000_050", "label": "person", "confidence": 0.9, "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0, "model_source": "yolo"}],
        }

    def table_columns(self, table):
        return self.COLUMNS.get(table, {})

    def sample_records(self, table, fields, limit):
        return tuple({field: row[field] for field in fields} for row in self.rows.get(table, ())[:limit])

    def get_frames_by_ids(self, frame_ids):
        return {frame_id: self.metadata[frame_id] for frame_id in frame_ids if frame_id in self.metadata}


class ContractValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.milvus = FakeMilvusInspector()
        self.es = FakeElasticsearchInspector()
        self.sqlite = FakeSQLiteInspector()

    def validator(self):
        return OfflineContractValidator(
            OnlineDataConfig(),
            milvus=self.milvus,
            elasticsearch=self.es,
            sqlite=self.sqlite,
            encoder_smoke_vectors={
                "visual_features": lambda: (1.0, 0.0),
                "ocr_features": lambda: (1.0, 0.0),
                "asr_features": lambda: (1.0, 0.0),
                "summary_features": lambda: (1.0, 0.0),
            },
        )

    def test_pass_for_consistent_complete_fixture(self) -> None:
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.PASS)
        self.assertEqual(report.dimensions["visual_features"], 2)
        self.assertIn("milvus:visual_features", report.resources_checked)
        self.assertEqual(report.sample_counts["milvus:visual"], 1)
        self.assertEqual(report.checks_skipped, ())

    def test_fail_for_core_frame_id_contract_mismatch(self) -> None:
        self.milvus.records["visual_features"][0]["frame_id"] = "shot_00000_pos_050"
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertTrue(
            any(check.name == "canonical_id.milvus.visual" for check in report.failed_checks)
        )

    def test_same_malformed_id_across_resources_still_fails(self) -> None:
        malformed = "shot_00000_pos_050"
        self.milvus.records["visual_features"][0]["frame_id"] = malformed
        self.sqlite.metadata = {
            malformed: FrameMetadata(
                frame_id=malformed,
                video_id="V1",
                shot_id=0,
                timestamp_sec=1.0,
            )
        }
        self.sqlite.rows["metadata"][0]["frame_id"] = malformed
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertTrue(
            any(check.name == "canonical_id.milvus.visual" for check in report.failed_checks)
        )

    def test_missing_encoder_smoke_is_explicit_not_run_and_not_pass(self) -> None:
        validator = OfflineContractValidator(
            OnlineDataConfig(),
            milvus=self.milvus,
            elasticsearch=self.es,
            sqlite=self.sqlite,
        )
        report = validator.validate()
        self.assertNotEqual(report.status, ValidationStatus.PASS)
        smoke = next(check for check in report.checks if check.name == "encoder.visual_features")
        self.assertEqual(smoke.status.value, "NOT_RUN")
        self.assertIn("encoder.visual_features", report.checks_skipped)

    def test_partial_when_optional_resource_is_missing(self) -> None:
        del self.milvus.records["ocr_features"]
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.PARTIAL)
        smoke = next(
            check for check in report.checks if check.name == "encoder.ocr_features"
        )
        self.assertEqual(smoke.status, CheckStatus.NOT_RUN)

    def test_backend_exception_still_lists_every_resource_subcheck(self) -> None:
        def fail_exists(name):
            if name == "visual_features":
                raise ConnectionError("backend unavailable")
            return name in self.milvus.records

        self.milvus.collection_exists = fail_exists
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        checks = {check.name: check for check in report.checks}
        self.assertEqual(checks["milvus.visual_features"].status, CheckStatus.FAIL)
        for suffix in (
            "exists",
            "fields",
            "types",
            "dimension",
            "index",
            "non_empty",
            "vector_norm",
        ):
            self.assertEqual(
                checks[f"milvus.visual_features.{suffix}"].status,
                CheckStatus.NOT_RUN,
            )
        self.assertEqual(
            checks["encoder.visual_features"].status,
            CheckStatus.NOT_RUN,
        )


if __name__ == "__main__":
    unittest.main()
