"""Verify a complete Offline dataset and atomically publish its READY manifest."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from verify_frame_id_consistency import (
    build_full_contract_report,
    collect_full_snapshot,
    record_counts,
)

from .dataset_manifest import (
    build_manifest_draft,
    publish_ready_manifest,
    write_manifest_draft,
)


DEFAULT_PRODUCER_CONFIG = {
    "module1": {
        "keyframe_positions": [0.15, 0.5, 0.85],
        "shot_threshold": 0.5,
        "webp_quality": 90,
    },
    "module2": {"precision": "fp16"},
    "module3": {
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "llm_provider": "local",
        "whisper_size": "medium",
    },
    "module3_intervals": {
        "max_interval_sec": 60.0,
        "min_interval_sec": 20.0,
        "target_interval_sec": 40.0,
    },
    "module4": {
        "confidence_threshold": 0.4,
        "mag_ratio": 1.5,
        "vietocr_backbone": "vgg_transformer",
        "width_ths": 0.7,
    },
    "module5": {
        "confidence_threshold": 0.25,
        "detector": "yolo-world",
        "model": "yolov8s-world.pt",
        "nms_threshold": 0.5,
    },
    "module6": {
        "asr_ocr_pooling": "direct_l2",
        "summary_pooling": "chunk_mean_l2",
    },
}


def _load_producer_config(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return DEFAULT_PRODUCER_CONFIG
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("producer config JSON must contain an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed dataset publication CLI parser."""
    parser = argparse.ArgumentParser(
        description="Verify and publish a self-indexed-v2 READY manifest"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data/processed")),
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--milvus-uri",
        default=os.getenv("MILVUS_URI", "http://localhost:19530"),
    )
    parser.add_argument(
        "--es-uri",
        default=os.getenv("ES_URI", "http://localhost:9200"),
    )
    parser.add_argument(
        "--db-uri",
        default=os.getenv("DB_URI", "sqlite:///data/metadata.db"),
    )
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--building-manifest-path", type=Path)
    parser.add_argument("--producer-config-json", type=Path)
    return parser


def main() -> int:
    """Return non-zero unless a fully verified READY manifest was published."""
    args = build_parser().parse_args()
    manifest_path = args.manifest_path or (
        args.data_dir / "dataset-manifest.json"
    )
    building_path = args.building_manifest_path or (
        args.data_dir / "dataset-manifest.building.json"
    )

    try:
        snapshot = collect_full_snapshot(
            milvus_uri=args.milvus_uri,
            es_uri=args.es_uri,
            db_uri=args.db_uri,
        )
        draft = build_manifest_draft(
            data_dir=args.data_dir,
            dataset_id=args.dataset_id,
            record_counts=record_counts(snapshot),
            producer_config=_load_producer_config(
                args.producer_config_json
            ),
        )
        write_manifest_draft(draft, building_path)
        errors = build_full_contract_report(
            snapshot,
            data_root=args.data_dir,
            manifest=draft.model_dump(),
        )
    except Exception as exc:
        print(f"Dataset verification could not complete: {exc}")
        return 1

    if errors:
        print("Dataset verification FAILED; READY was not published:")
        for error in errors:
            print(f"- {error}")
        return 1

    try:
        ready = publish_ready_manifest(
            draft,
            manifest_path,
            verification_errors=errors,
        )
    except Exception as exc:
        print(f"READY manifest publication failed: {exc}")
        return 1

    print("Dataset verification PASSED and READY was published.")
    print(f"dataset_id={ready.dataset_id}")
    print(f"dataset_fingerprint={ready.dataset_fingerprint}")
    print(f"manifest_path={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
