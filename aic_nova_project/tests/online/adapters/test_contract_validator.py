from __future__ import annotations

import unittest

from online.adapters.contract_validator import (
    CheckStatus,
    OfflineContractValidator,
    ValidationStatus,
)
from online.config import OnlineDataConfig
from online.ports.manifest import DatasetManifest
from online.ports.records import FrameMetadata

FRAME_ID = "L21_V001_001"
VIDEO_ID = "L21_V001"
UNIT_512 = (1.0,) + (0.0,) * 511


class FakeManifestReader:
    def __init__(self, model_id: str = "ViT-B-32::openai") -> None:
        self.model_id = model_id

    def read_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            contract_version="organizer-v1",
            visual_model_id=self.model_id,
            visual_dimension=512,
            visual_normalized=True,
            frame_id_contract_version="organizer-v1",
            object_threshold=0.25,
            object_nms_iou=0.45,
            record_counts={"metadata": 1},
            dataset_fingerprint="fixture-fingerprint",
        )


class FakeMilvusInspector:
    def __init__(self) -> None:
        self.records = {
            "visual_features": [
                {
                    "frame_id": FRAME_ID,
                    "video_id": VIDEO_ID,
                    "local_index": 0,
                    "embedding": UNIT_512,
                }
            ],
            "ocr_features": [
                {"frame_id": FRAME_ID, "video_id": VIDEO_ID, "embedding": UNIT_512}
            ],
            "asr_features": [
                {
                    "video_id": VIDEO_ID,
                    "interval_id": "i1",
                    "start_time_sec": 0.0,
                    "end_time_sec": 2.0,
                    "embedding": UNIT_512,
                }
            ],
            "summary_features": [{"video_id": VIDEO_ID, "embedding": UNIT_512}],
        }
        self.fields = {
            "visual_features": {
                "frame_id": "VARCHAR",
                "video_id": "VARCHAR",
                "local_index": "INT64",
                "embedding": "FLOAT_VECTOR",
            },
            "ocr_features": {
                "frame_id": "VARCHAR",
                "video_id": "VARCHAR",
                "embedding": "FLOAT_VECTOR",
            },
            "asr_features": {
                "video_id": "VARCHAR",
                "interval_id": "VARCHAR",
                "start_time_sec": "FLOAT",
                "end_time_sec": "FLOAT",
                "embedding": "FLOAT_VECTOR",
            },
            "summary_features": {"video_id": "VARCHAR", "embedding": "FLOAT_VECTOR"},
        }

    def collection_exists(self, name):
        return name in self.records

    def describe_collection(self, name):
        return {
            "fields": self.fields[name],
            "dimension": 512,
            "metric_type": "IP",
            "index_type": "HNSW",
        }

    def sample_records(self, name, output_fields, limit):
        return tuple(
            {field: row[field] for field in output_fields}
            for row in self.records[name][:limit]
        )


class FakeElasticsearchInspector:
    TYPES = {
        "ocr_texts": {
            "frame_id": "keyword",
            "video_id": "keyword",
            "ocr_text_concat": "text",
        },
        "asr_transcripts": {
            "video_id": "keyword",
            "interval_id": "keyword",
            "start_time_sec": "float",
            "end_time_sec": "float",
            "cleaned_text": "text",
        },
        "video_summaries": {"video_id": "keyword", "summary": "text"},
    }

    def __init__(self) -> None:
        self.documents = {
            "ocr_texts": [
                {"frame_id": FRAME_ID, "video_id": VIDEO_ID, "ocr_text_concat": "text"}
            ],
            "asr_transcripts": [
                {
                    "video_id": VIDEO_ID,
                    "interval_id": "i1",
                    "start_time_sec": 0.0,
                    "end_time_sec": 2.0,
                    "cleaned_text": "speech",
                }
            ],
            "video_summaries": [{"video_id": VIDEO_ID, "summary": "summary"}],
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

    def sample_documents(self, name, fields, limit):
        return tuple(
            {field: row[field] for field in fields}
            for row in self.documents[name][:limit]
        )

    def find_documents(self, name, filters, fields, limit=2):
        return tuple(
            {field: row[field] for field in fields}
            for row in self.documents[name]
            if all(row.get(field) == value for field, value in filters.items())
        )[:limit]

    def has_icu_plugin(self):
        return True


class FakeSQLiteInspector:
    COLUMNS = {
        "videos": {
            "video_id": "TEXT",
            "media_title": "TEXT",
            "media_author": "TEXT",
            "media_description": "TEXT",
            "media_keywords_json": "TEXT",
            "media_length_sec": "REAL",
            "publish_date": "TEXT",
            "watch_url": "TEXT",
            "video_rel_path": "TEXT",
        },
        "metadata": {
            "frame_id": "TEXT",
            "video_id": "TEXT",
            "keyframe_no": "INTEGER",
            "local_index": "INTEGER",
            "pts_time_sec": "REAL",
            "fps": "REAL",
            "source_frame_idx": "INTEGER",
            "image_rel_path": "TEXT",
        },
        "objects": {
            "id": "INTEGER",
            "frame_id": "TEXT",
            "label_display": "TEXT",
            "label_normalized": "TEXT",
            "class_mid": "TEXT",
            "class_label_id": "TEXT",
            "confidence": "REAL",
            "x_min": "REAL",
            "y_min": "REAL",
            "x_max": "REAL",
            "y_max": "REAL",
            "model_source": "TEXT",
        },
    }

    def __init__(self) -> None:
        self.metadata = {
            FRAME_ID: FrameMetadata(
                frame_id=FRAME_ID,
                video_id=VIDEO_ID,
                keyframe_no=1,
                local_index=0,
                timestamp_sec=1.0,
                fps=25.0,
                source_frame_idx=25,
                image_rel_path="L21_V001/001.jpg",
            )
        }
        self.rows = {
            "videos": [
                {
                    "video_id": VIDEO_ID,
                    "media_title": None,
                    "media_author": None,
                    "media_description": None,
                    "media_keywords_json": None,
                    "media_length_sec": None,
                    "publish_date": None,
                    "watch_url": None,
                    "video_rel_path": "videos/L21_V001.mp4",
                }
            ],
            "metadata": [
                {
                    "frame_id": FRAME_ID,
                    "video_id": VIDEO_ID,
                    "keyframe_no": 1,
                    "local_index": 0,
                    "pts_time_sec": 1.0,
                    "fps": 25.0,
                    "source_frame_idx": 25,
                    "image_rel_path": "L21_V001/001.jpg",
                }
            ],
            "objects": [
                {
                    "id": 1,
                    "frame_id": FRAME_ID,
                    "label_display": "Person",
                    "label_normalized": "person",
                    "class_mid": "/m/01g317",
                    "class_label_id": None,
                    "confidence": 0.9,
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 1.0,
                    "y_max": 1.0,
                    "model_source": "yolo",
                }
            ],
        }

    def table_columns(self, table):
        return self.COLUMNS.get(table, {})

    def sample_records(self, table, fields, limit):
        return tuple(
            {field: row[field] for field in fields}
            for row in self.rows.get(table, ())[:limit]
        )

    def get_frames_by_ids(self, frame_ids):
        return {
            frame_id: self.metadata[frame_id]
            for frame_id in frame_ids
            if frame_id in self.metadata
        }


class ContractValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.milvus, self.es, self.sqlite, self.manifest = (
            FakeMilvusInspector(),
            FakeElasticsearchInspector(),
            FakeSQLiteInspector(),
            FakeManifestReader(),
        )

    def validator(self):
        return OfflineContractValidator(
            OnlineDataConfig(),
            milvus=self.milvus,
            elasticsearch=self.es,
            sqlite=self.sqlite,
            manifest=self.manifest,
            encoder_smoke_vectors={
                name: lambda: UNIT_512
                for name in (
                    "visual_features",
                    "ocr_features",
                    "asr_features",
                    "summary_features",
                )
            },
        )

    def test_pass_for_consistent_complete_organizer_fixture(self) -> None:
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.PASS)
        self.assertEqual(report.dimensions["visual_features"], 512)
        self.assertIn("manifest:dataset_manifest", report.resources_checked)
        self.assertEqual(report.checks_skipped, ())

    def test_wrong_visual_model_fails_even_with_dimension_512(self) -> None:
        self.manifest.model_id = "ViT-L-14::openai"
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertEqual(
            next(
                c for c in report.checks if c.name == "manifest.visual_model_id"
            ).status,
            CheckStatus.FAIL,
        )

    def test_canonical_id_or_local_index_mismatch_fails(self) -> None:
        self.milvus.records["visual_features"][0]["local_index"] = 3
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertTrue(
            any(c.name == "canonical_id.milvus.visual" for c in report.failed_checks)
        )

    def test_missing_encoder_smoke_is_explicit_not_run(self) -> None:
        report = OfflineContractValidator(
            OnlineDataConfig(),
            milvus=self.milvus,
            elasticsearch=self.es,
            sqlite=self.sqlite,
            manifest=self.manifest,
        ).validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertEqual(
            next(
                c for c in report.checks if c.name == "encoder.visual_features"
            ).status,
            CheckStatus.NOT_RUN,
        )

    def test_backend_error_diagnostic_does_not_leak_secret(self) -> None:
        class BrokenManifest:
            def read_manifest(self):
                raise OSError("postgres://admin:secret@host/private/path")

        self.manifest = BrokenManifest()
        dumped = self.validator().validate().model_dump_json()
        self.assertNotIn("secret", dumped)
        self.assertNotIn("private/path", dumped)
        self.assertIn("OSError", dumped)

    def test_optional_resource_missing_is_partial(self) -> None:
        del self.milvus.records["ocr_features"]
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.PARTIAL)
        self.assertEqual(
            next(c for c in report.checks if c.name == "encoder.ocr_features").status,
            CheckStatus.NOT_RUN,
        )


if __name__ == "__main__":
    unittest.main()
