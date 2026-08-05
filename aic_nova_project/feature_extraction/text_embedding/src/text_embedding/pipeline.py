import logging
from pathlib import Path
from typing import Optional

from .encoders import BaseTextEncoder
from .data_readers import parse_asr_file, parse_summary_file, parse_ocr_file
from .embedding_writer import write_embeddings_to_parquet
from .artifact_contract import (
    add_artifact_contract,
    build_encoder_provenance,
    is_valid_text_embedding_artifact,
    source_sha256,
)

logger = logging.getLogger(__name__)

class TextEmbeddingPipeline:
    def __init__(self, encoder: BaseTextEncoder):
        self.encoder = encoder

    def process_asr(self, input_dir: Path, output_dir: Path, force: bool = False):
        """Processes ASR JSONs to Parquet embeddings using batch encoding."""
        output_dir.mkdir(parents=True, exist_ok=True)
        seen_video_ids = set()
        for json_path in sorted(input_dir.glob("*_cleaned.json")):
            records = parse_asr_file(json_path)
            if not records:
                logger.debug(f"No valid ASR records in {json_path}")
                continue

            video_id = records[0]["video_id"]
            if video_id in seen_video_ids:
                raise ValueError(
                    f"Duplicate ASR artifact for video_id {video_id}."
                )
            seen_video_ids.add(video_id)
            output_path = output_dir / f"{video_id}.parquet"
            fingerprint = source_sha256(json_path)
            provenance = build_encoder_provenance(self.encoder, "asr")
            if (
                output_path.exists()
                and not force
                and is_valid_text_embedding_artifact(
                    output_path,
                    expected_records=records,
                    artifact_kind="asr",
                    source_fingerprint=fingerprint,
                    provenance=provenance,
                )
            ):
                logger.info(f"Skipping {video_id} (ASR) - artifact is valid.")
                continue

            texts = [r["text"] for r in records]
            embeddings = self.encoder.encode_batch(texts)
            if len(embeddings) != len(records):
                raise RuntimeError(
                    f"ASR encoder returned {len(embeddings)} vectors for "
                    f"{len(records)} records in {json_path}."
                )

            for i, record in enumerate(records):
                record["embedding"] = embeddings[i].tolist()

            add_artifact_contract(
                records,
                artifact_kind="asr",
                source_fingerprint=fingerprint,
                provenance=provenance,
            )
            write_embeddings_to_parquet(records, output_path)

    def process_ocr(self, input_dir: Path, output_dir: Path, force: bool = False):
        """Processes OCR JSONs to Parquet embeddings using batch encoding."""
        output_dir.mkdir(parents=True, exist_ok=True)
        seen_video_ids = set()
        for json_path in sorted(input_dir.glob("*.json")):
            records = parse_ocr_file(json_path)
            if not records:
                logger.debug(f"No valid OCR records in {json_path}")
                continue

            video_id = records[0]["video_id"]
            if video_id in seen_video_ids:
                raise ValueError(
                    f"Duplicate OCR artifact for video_id {video_id}."
                )
            seen_video_ids.add(video_id)
            output_path = output_dir / f"{video_id}.parquet"
            fingerprint = source_sha256(json_path)
            provenance = build_encoder_provenance(self.encoder, "ocr")
            if (
                output_path.exists()
                and not force
                and is_valid_text_embedding_artifact(
                    output_path,
                    expected_records=records,
                    artifact_kind="ocr",
                    source_fingerprint=fingerprint,
                    provenance=provenance,
                )
            ):
                logger.info(f"Skipping {video_id} (OCR) - artifact is valid.")
                continue

            texts = [r["text"] for r in records]
            embeddings = self.encoder.encode_batch(texts)
            if len(embeddings) != len(records):
                raise RuntimeError(
                    f"OCR encoder returned {len(embeddings)} vectors for "
                    f"{len(records)} records in {json_path}."
                )

            for i, record in enumerate(records):
                record["embedding"] = embeddings[i].tolist()

            add_artifact_contract(
                records,
                artifact_kind="ocr",
                source_fingerprint=fingerprint,
                provenance=provenance,
            )
            write_embeddings_to_parquet(records, output_path)

    def process_summary(self, input_dir: Path, output_dir: Path, force: bool = False):
        """Processes Summary JSONs to Parquet embeddings using chunking & mean-pooling."""
        output_dir.mkdir(parents=True, exist_ok=True)
        seen_video_ids = set()
        for json_path in sorted(input_dir.glob("*.json")):
            records = parse_summary_file(json_path)
            if not records:
                logger.debug(f"No valid Summary records in {json_path}")
                continue

            video_id = records[0]["video_id"]
            if video_id in seen_video_ids:
                raise ValueError(
                    f"Duplicate summary artifact for video_id {video_id}."
                )
            seen_video_ids.add(video_id)
            output_path = output_dir / f"{video_id}.parquet"
            fingerprint = source_sha256(json_path)
            provenance = build_encoder_provenance(
                self.encoder,
                "summary",
            )
            if (
                output_path.exists()
                and not force
                and is_valid_text_embedding_artifact(
                    output_path,
                    expected_records=records,
                    artifact_kind="summary",
                    source_fingerprint=fingerprint,
                    provenance=provenance,
                )
            ):
                logger.info(
                    f"Skipping {video_id} (Summary) - artifact is valid."
                )
                continue

            # Summary uses encode_long_text for chunking + mean pooling
            for record in records:
                embedding = self.encoder.encode_long_text(record["text"])
                record["embedding"] = embedding.tolist()

            add_artifact_contract(
                records,
                artifact_kind="summary",
                source_fingerprint=fingerprint,
                provenance=provenance,
            )
            write_embeddings_to_parquet(records, output_path)
