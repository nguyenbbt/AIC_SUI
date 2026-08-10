from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from online.adapters.contract_validator import (
    CheckStatus,
    OfflineContractValidator,
    ValidationStatus,
)
from online.config import DatasetResourceConfig, OnlineDataConfig
from online.domain.manifest import DatasetManifest
from online.ports.records import FrameMetadata, VideoMetadata


class FakeMilvusInspector:
    def __init__(self) -> None:
        self.records = {
            "visual_features": [{"frame_id": "V1_00000_050", "video_id": "V1", "shot_id": 0, "embedding": [1.0, 0.0]}],
            "ocr_features": [{"frame_id": "V1_00000_050", "video_id": "V1", "embedding": [1.0, 0.0]}],
            "asr_features": [{"video_id": "V1", "interval_id": "0", "start_time_sec": 0.0, "end_time_sec": 2.0, "embedding": [1.0, 0.0]}],
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

    def iter_records(self, name, output_fields, *, filter_expression, batch_size):
        del filter_expression
        for start in range(0, len(self.records[name]), batch_size):
            yield tuple(
                {field: record[field] for field in output_fields}
                for record in self.records[name][start : start + batch_size]
            )


class FakeElasticsearchInspector:
    TYPES = {
        "ocr_texts": {"frame_id": "keyword", "video_id": "keyword", "shot_id": "keyword", "ocr_text_concat": "text"},
        "asr_transcripts": {"video_id": "keyword", "interval_id": "keyword", "start_time_sec": "float", "end_time_sec": "float", "cleaned_text": "text"},
        "video_summaries": {"video_id": "keyword", "summary": "text"},
    }

    def __init__(self) -> None:
        self.documents = {
            "ocr_texts": [{"frame_id": "V1_00000_050", "video_id": "V1", "shot_id": "0", "ocr_text_concat": "text"}],
            "asr_transcripts": [{"video_id": "V1", "interval_id": "0", "start_time_sec": 0.0, "end_time_sec": 2.0, "cleaned_text": "speech"}],
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

    def iter_documents(self, name, source_fields, *, batch_size):
        for start in range(0, len(self.documents[name]), batch_size):
            yield tuple(
                {field: document[field] for field in source_fields}
                for document in self.documents[name][start : start + batch_size]
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
        "videos": {"video_id": "TEXT", "source_video_rel_path": "TEXT", "fps": "REAL", "duration_sec": "REAL", "frame_count": "INTEGER", "width": "INTEGER", "height": "INTEGER"},
        "metadata": {"frame_id": "TEXT", "video_id": "TEXT", "shot_id": "INTEGER", "source_frame_idx": "INTEGER", "timestamp": "REAL", "image_rel_path": "TEXT"},
        "objects": {"id": "INTEGER", "frame_id": "TEXT", "label": "TEXT", "confidence": "REAL", "x_min": "REAL", "y_min": "REAL", "x_max": "REAL", "y_max": "REAL", "model_source": "TEXT"},
    }

    def __init__(self) -> None:
        self.metadata = {
            "V1_00000_050": FrameMetadata(frame_id="V1_00000_050", video_id="V1", shot_id=0, source_frame_idx=30, timestamp_sec=1.0, image_rel_path="keyframes/V1/V1_00000_050.jpg")
        }
        self.videos = {
            "V1": VideoMetadata(video_id="V1", source_video_rel_path="videos/V1.mp4", fps=30.0, duration_sec=2.0, frame_count=60, width=1920, height=1080)
        }
        self.rows = {
            "videos": [{"video_id": "V1", "source_video_rel_path": "videos/V1.mp4", "fps": 30.0, "duration_sec": 2.0, "frame_count": 60, "width": 1920, "height": 1080}],
            "metadata": [{"frame_id": "V1_00000_050", "video_id": "V1", "shot_id": 0, "source_frame_idx": 30, "timestamp": 1.0, "image_rel_path": "keyframes/V1/V1_00000_050.jpg"}],
            "objects": [{"id": 1, "frame_id": "V1_00000_050", "label": "person", "confidence": 0.9, "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0, "model_source": "yolo"}],
        }

    def table_columns(self, table):
        return self.COLUMNS.get(table, {})

    def sample_records(self, table, fields, limit):
        return tuple({field: row[field] for field in fields} for row in self.rows.get(table, ())[:limit])

    def iter_records(self, table, fields, *, batch_size):
        records = self.rows.get(table, ())
        for start in range(0, len(records), batch_size):
            yield tuple(
                {field: row[field] for field in fields}
                for row in records[start : start + batch_size]
            )

    def get_frames_by_ids(self, frame_ids):
        return {frame_id: self.metadata[frame_id] for frame_id in frame_ids if frame_id in self.metadata}

    def get_videos_by_ids(self, video_ids):
        return {video_id: self.videos[video_id] for video_id in video_ids if video_id in self.videos}


class FakeManifestGate:
    def __init__(self, manifest: DatasetManifest) -> None:
        self.manifest = manifest

    def health_check(self) -> None:
        return None


class ContractValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.milvus = FakeMilvusInspector()
        self.es = FakeElasticsearchInspector()
        self.sqlite = FakeSQLiteInspector()

    def validator(
        self,
        *,
        config: OnlineDataConfig | None = None,
        manifest_gate=None,
        sample_size: int = 5,
    ):
        return OfflineContractValidator(
            config
            or OnlineDataConfig(
                dataset=DatasetResourceConfig(manifest_required=False)
            ),
            milvus=self.milvus,
            elasticsearch=self.es,
            sqlite=self.sqlite,
            manifest_gate=manifest_gate,
            encoder_smoke_vectors={
                "visual_features": lambda: (1.0, 0.0),
                "ocr_features": lambda: (1.0, 0.0),
                "asr_features": lambda: (1.0, 0.0),
                "summary_features": lambda: (1.0, 0.0),
            },
            sample_size=sample_size,
        )

    def add_second_frame(self) -> str:
        frame_id = "V1_00001_050"
        metadata = FrameMetadata(
            frame_id=frame_id,
            video_id="V1",
            shot_id=1,
            source_frame_idx=45,
            timestamp_sec=1.5,
            image_rel_path=f"keyframes/V1/{frame_id}.jpg",
        )
        self.sqlite.metadata[frame_id] = metadata
        self.sqlite.rows["metadata"].append(
            {
                "frame_id": frame_id,
                "video_id": "V1",
                "shot_id": 1,
                "source_frame_idx": 45,
                "timestamp": 1.5,
                "image_rel_path": metadata.image_rel_path,
            }
        )
        self.milvus.records["visual_features"].append(
            {
                "frame_id": frame_id,
                "video_id": "V1",
                "shot_id": 1,
                "embedding": [1.0, 0.0],
            }
        )
        return frame_id

    def test_pass_for_consistent_complete_fixture(self) -> None:
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.PASS)
        self.assertEqual(report.dimensions["visual_features"], 2)
        self.assertIn("milvus:visual_features", report.resources_checked)
        self.assertEqual(report.sample_counts["milvus:visual"], 1)
        self.assertEqual(report.actual_counts["visual_features"], 1)
        self.assertEqual(report.audit_scope, "FULL")
        self.assertEqual(report.checks_skipped, ())

    def test_analyzer_audit_ignores_non_contract_text_fields(self) -> None:
        original_get_mapping = self.es.get_mapping

        def mapping_with_internal_field(name):
            mapping = original_get_mapping(name)
            if name == "asr_transcripts":
                mapping["properties"]["_doc_id"] = {"type": "text"}
            return mapping

        self.es.get_mapping = mapping_with_internal_field

        report = self.validator().validate()
        analyzer_check = next(
            check
            for check in report.checks
            if check.name == "elasticsearch.asr_transcripts.analyzer"
        )

        self.assertEqual(analyzer_check.status, CheckStatus.PASS)
        self.assertEqual(report.status, ValidationStatus.PASS)

    def test_fail_for_core_frame_id_contract_mismatch(self) -> None:
        self.milvus.records["visual_features"][0]["frame_id"] = "shot_00000_pos_050"
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertTrue(
            any(check.name == "canonical_id.milvus.visual" for check in report.failed_checks)
        )

    def test_fail_when_metadata_source_frame_exceeds_video_bounds(self) -> None:
        self.sqlite.metadata["V1_00000_050"] = self.sqlite.metadata[
            "V1_00000_050"
        ].model_copy(update={"source_frame_idx": 60})
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertTrue(
            any(check.name == "join.metadata_to_videos" for check in report.failed_checks)
        )

    def test_object_bbox_outside_video_is_reported_as_optional_warning(self) -> None:
        self.sqlite.rows["objects"][0]["x_max"] = 2000.0
        report = self.validator().validate()
        self.assertEqual(report.status, ValidationStatus.PARTIAL)
        check = next(
            check for check in report.checks if check.name == "join.objects_to_metadata"
        )
        self.assertEqual(check.status, CheckStatus.WARNING)

    def test_same_malformed_id_across_resources_still_fails(self) -> None:
        malformed = "shot_00000_pos_050"
        self.milvus.records["visual_features"][0]["frame_id"] = malformed
        self.sqlite.metadata = {
            malformed: FrameMetadata.model_construct(
                frame_id=malformed,
                video_id="V1",
                shot_id=0,
                timestamp_sec=1.0,
                source_frame_idx=30,
                image_rel_path="keyframes/V1/malformed.jpg",
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
            OnlineDataConfig(dataset=DatasetResourceConfig(manifest_required=False)),
            milvus=self.milvus,
            elasticsearch=self.es,
            sqlite=self.sqlite,
        )
        report = validator.validate()
        self.assertNotEqual(report.status, ValidationStatus.PASS)
        smoke = next(check for check in report.checks if check.name == "encoder.visual_features")
        self.assertEqual(smoke.status.value, "NOT_RUN")
        self.assertIn("encoder.visual_features", report.checks_skipped)

    def test_required_manifest_missing_blocks_full_audit_scope(self) -> None:
        validator = OfflineContractValidator(
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

        report = validator.validate()

        check = next(
            check
            for check in report.checks
            if check.name == "full.manifest.record_counts"
        )
        self.assertEqual(check.status, CheckStatus.NOT_RUN)
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertEqual(report.audit_scope, "INCOMPLETE")

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

    def test_full_scan_finds_invalid_vector_after_clean_sample(self) -> None:
        self.add_second_frame()
        self.milvus.records["visual_features"][-1]["embedding"] = [0.0, 0.0]

        report = self.validator(sample_size=1).validate()

        checks = {check.name: check for check in report.checks}
        self.assertEqual(
            checks["milvus.visual_features.vector_norm"].status,
            CheckStatus.PASS,
        )
        self.assertEqual(
            checks["full.milvus.visual_features"].status,
            CheckStatus.FAIL,
        )
        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertEqual(report.audit_scope, "INCOMPLETE")

    def test_full_digest_detects_equal_count_different_ocr_key_sets(self) -> None:
        second = self.add_second_frame()
        self.es.documents["ocr_texts"] = [
            {
                "frame_id": second,
                "video_id": "V1",
                "shot_id": "1",
                "ocr_text_concat": "different frame",
            }
        ]

        report = self.validator(sample_size=1).validate()

        check = next(
            check
            for check in report.checks
            if check.name == "full.join.ocr_dense_to_lexical"
        )
        self.assertEqual(check.status, CheckStatus.WARNING)
        self.assertEqual(report.status, ValidationStatus.PARTIAL)

    def test_full_scan_rejects_duplicate_domain_key(self) -> None:
        self.milvus.records["visual_features"].append(
            dict(self.milvus.records["visual_features"][0])
        )

        report = self.validator(sample_size=1).validate()

        check = next(
            check
            for check in report.checks
            if check.name == "full.milvus.visual_features"
        )
        self.assertEqual(check.status, CheckStatus.FAIL)
        self.assertIn("check could not complete", check.message)

    def test_missing_full_scan_capability_cannot_report_pass(self) -> None:
        self.milvus.iter_records = None

        report = self.validator().validate()

        self.assertEqual(report.status, ValidationStatus.FAIL)
        self.assertEqual(report.audit_scope, "INCOMPLETE")

    def test_manifest_counts_are_compared_with_actual_full_scan_counts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "videos").mkdir()
            (root / "videos" / "V1.mp4").write_bytes(b"video")
            image_path = root / "keyframes" / "V1" / "V1_00000_050.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2), color="white").save(image_path)
            counts = {
                "videos": 1,
                "metadata": 1,
                "objects": 2,
                "visual_features": 1,
                "ocr_features": 1,
                "asr_features": 1,
                "summary_features": 1,
                "ocr_texts": 1,
                "asr_transcripts": 1,
                "video_summaries": 1,
            }
            manifest = DatasetManifest.model_validate(
                {
                    "contract_version": "self-indexed-v2",
                    "dataset_id": "fixture-run",
                    "dataset_fingerprint": "sha256:" + "a" * 64,
                    "status": "READY",
                    "frame_index_base": 0,
                    "bbox_space": "absolute_pixel_xyxy",
                    "visual_model_id": "ViT-B-32::openai",
                    "visual_dimension": 512,
                    "visual_normalized": True,
                    "text_model_name": "dangvantuan/vietnamese-embedding",
                    "text_model_revision": "4ab46e46ba5902328ba0742e489e75f787932f2b",
                    "text_dimension": 768,
                    "text_max_length": 256,
                    "record_counts": counts,
                    "created_at_utc": "2026-08-05T00:00:00Z",
                }
            )
            config = OnlineDataConfig(
                dataset=DatasetResourceConfig(
                    data_root=root,
                    manifest_required=True,
                )
            )

            report = self.validator(
                config=config,
                manifest_gate=FakeManifestGate(manifest),
            ).validate()

        check = next(
            check
            for check in report.checks
            if check.name == "full.manifest.record_counts"
        )
        self.assertEqual(check.status, CheckStatus.FAIL)
        self.assertEqual(report.actual_counts["objects"], 1)
        self.assertEqual(report.status, ValidationStatus.FAIL)


if __name__ == "__main__":
    unittest.main()
