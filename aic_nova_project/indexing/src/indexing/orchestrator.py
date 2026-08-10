"""
Orchestrator: coordinates data ingestion into Milvus, Elasticsearch, and SQLite.

Implements per-video transactional semantics:
- Delete-then-Insert (idempotent upsert)
- Rollback on partial failure (if ES fails after Milvus succeeds,
  Milvus records for that video are rolled back)
- Graceful degradation (missing OCR/Object data is silently skipped)
"""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .clients.milvus_client import (
    MilvusVectorClient,
    VISUAL_COLLECTION,
    ASR_COLLECTION,
    SUMMARY_COLLECTION,
    OCR_COLLECTION,
)
from .clients.es_client import ESClient, OCR_INDEX, ASR_INDEX, SUMMARY_INDEX
from .clients.tabular_client import TabularClient
from .data_loader import (
    discover_video_ids,
    detect_embedding_dim,
    load_visual_embeddings,
    load_text_asr_embeddings,
    load_text_summary_embeddings,
    load_text_ocr_embeddings,
    load_ocr_texts,
    load_asr_transcripts,
    load_video_summary,
    load_video_metadata,
    load_metadata_and_objects,
)

logger = logging.getLogger(__name__)


@dataclass
class VideoSnapshot:
    """Restorable last-known-good state for one video across all backends."""

    milvus: Dict[str, list]
    elasticsearch: Dict[str, list]
    metadata: list
    objects: list
    video: Optional[dict] = None


class IndexingOrchestrator:
    """
    Coordinates the ingestion of all video data into 3 databases.
    Processes one video at a time with rollback on failure.
    """

    def __init__(
        self,
        milvus_client: MilvusVectorClient,
        es_client: ESClient,
        tabular_client: TabularClient,
        batch_size: int = 500,
    ):
        self.milvus = milvus_client
        self.es = es_client
        self.tabular = tabular_client
        self.batch_size = batch_size

    def _delete_video_from_all(self, video_id: str):
        """Delete all records for a video across all 3 DBs (pre-insert cleanup)."""
        # Milvus
        self.milvus.delete_by_video_id(VISUAL_COLLECTION, video_id)
        self.milvus.delete_by_video_id(ASR_COLLECTION, video_id)
        self.milvus.delete_by_video_id(SUMMARY_COLLECTION, video_id)
        self.milvus.delete_by_video_id(OCR_COLLECTION, video_id)

        # Elasticsearch
        self.es.delete_by_video_id(OCR_INDEX, video_id)
        self.es.delete_by_video_id(ASR_INDEX, video_id)
        self.es.delete_by_video_id(SUMMARY_INDEX, video_id)

        # SQLite
        self.tabular.delete_by_video_id(video_id)

    @staticmethod
    def _snapshot_records(value) -> list:
        """Normalize real client results while keeping legacy mocks harmless."""
        return value if isinstance(value, list) else []

    def _capture_snapshot(self, video_id: str) -> VideoSnapshot:
        """Capture every backend before the first destructive mutation."""
        milvus_snapshot = {
            collection: self._snapshot_records(
                self.milvus.snapshot_by_video_id(collection, video_id)
            )
            for collection in (
                VISUAL_COLLECTION,
                ASR_COLLECTION,
                SUMMARY_COLLECTION,
                OCR_COLLECTION,
            )
        }
        es_snapshot = {
            index_name: self._snapshot_records(
                self.es.snapshot_by_video_id(index_name, video_id)
            )
            for index_name in (OCR_INDEX, ASR_INDEX, SUMMARY_INDEX)
        }
        tabular_snapshot = self.tabular.snapshot_by_video_id(video_id)
        if (
            isinstance(tabular_snapshot, tuple)
            and len(tabular_snapshot) == 2
        ):
            metadata = self._snapshot_records(tabular_snapshot[0])
            objects = self._snapshot_records(tabular_snapshot[1])
        else:
            metadata = []
            objects = []
        video_snapshot = self.tabular.snapshot_video_by_id(video_id)
        video = video_snapshot if isinstance(video_snapshot, dict) else None

        return VideoSnapshot(
            milvus=milvus_snapshot,
            elasticsearch=es_snapshot,
            metadata=metadata,
            objects=objects,
            video=video,
        )

    def _restore_snapshot(
        self,
        video_id: str,
        snapshot: VideoSnapshot,
    ) -> None:
        """Remove partial writes and restore the captured cross-DB state."""
        self._delete_video_from_all(video_id)

        for collection_name, records in snapshot.milvus.items():
            if not records:
                continue
            dimension = len(records[0]["embedding"])
            self._insert_batched(
                records,
                lambda batch, name=collection_name, dim=dimension: (
                    self.milvus.insert_batch(name, batch, dim)
                ),
                self.batch_size,
            )

        for index_name, documents in snapshot.elasticsearch.items():
            if documents:
                self.es.restore_snapshot(index_name, documents)

        self.tabular.restore_video_snapshot(snapshot.video)
        self.tabular.restore_snapshot(
            snapshot.metadata,
            snapshot.objects,
        )

    @staticmethod
    def _same_record_keys(
        existing: list,
        expected: list,
        fields: tuple[str, ...],
    ) -> bool:
        """Compare record identity as a multiset, preserving duplicates."""
        def keys(records: list) -> Counter:
            return Counter(
                tuple(str(record.get(field, "")) for field in fields)
                for record in records
            )

        return keys(existing) == keys(expected)

    def _snapshot_matches_inputs(
        self,
        snapshot: VideoSnapshot,
        *,
        visual_records: list,
        asr_emb_records: list,
        summary_emb_records: list,
        ocr_emb_records: list,
        ocr_text_records: list,
        asr_text_records: list,
        summary_text_records: list,
        metadata_records: list,
        object_records: list,
        video_record: Optional[dict] = None,
        mismatch_details: Optional[List[str]] = None,
    ) -> bool:
        """Return whether every persisted stream has the expected identities."""
        es_sources = {
            index_name: [
                document.get("_source", {})
                for document in documents
            ]
            for index_name, documents in snapshot.elasticsearch.items()
        }
        comparisons = (
            (
                f"milvus.{VISUAL_COLLECTION}",
                snapshot.milvus[VISUAL_COLLECTION],
                visual_records,
                ("frame_id",),
            ),
            (
                f"milvus.{ASR_COLLECTION}",
                snapshot.milvus[ASR_COLLECTION],
                asr_emb_records,
                ("video_id", "interval_id"),
            ),
            (
                f"milvus.{SUMMARY_COLLECTION}",
                snapshot.milvus[SUMMARY_COLLECTION],
                summary_emb_records,
                ("video_id",),
            ),
            (
                f"milvus.{OCR_COLLECTION}",
                snapshot.milvus[OCR_COLLECTION],
                ocr_emb_records,
                ("frame_id",),
            ),
            (
                f"elasticsearch.{OCR_INDEX}",
                es_sources[OCR_INDEX],
                ocr_text_records,
                ("frame_id",),
            ),
            (
                f"elasticsearch.{ASR_INDEX}",
                es_sources[ASR_INDEX],
                asr_text_records,
                ("video_id", "interval_id"),
            ),
            (
                f"elasticsearch.{SUMMARY_INDEX}",
                es_sources[SUMMARY_INDEX],
                summary_text_records,
                ("video_id",),
            ),
            (
                "sqlite.metadata",
                snapshot.metadata,
                metadata_records,
                ("frame_id",),
            ),
            (
                "sqlite.objects",
                snapshot.objects,
                object_records,
                (
                    "frame_id",
                    "label",
                    "confidence",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "model_source",
                ),
            ),
        )
        if video_record is not None:
            comparisons = comparisons + (
                (
                    "sqlite.videos",
                    [snapshot.video] if snapshot.video is not None else [],
                    [video_record],
                    (
                        "video_id",
                        "source_video_rel_path",
                        "fps",
                        "duration_sec",
                        "frame_count",
                        "width",
                        "height",
                    ),
                ),
            )
        matches = True
        for stream_name, existing, expected, fields in comparisons:
            if self._same_record_keys(existing, expected, fields):
                continue

            matches = False
            if mismatch_details is None:
                continue

            existing_keys = Counter(
                tuple(str(record.get(field, "")) for field in fields)
                for record in existing
            )
            expected_keys = Counter(
                tuple(str(record.get(field, "")) for field in fields)
                for record in expected
            )
            missing = list((expected_keys - existing_keys).elements())[:3]
            unexpected = list((existing_keys - expected_keys).elements())[:3]
            mismatch_details.append(
                f"{stream_name}: expected={len(expected)} "
                f"actual={len(existing)} missing={missing!r} "
                f"unexpected={unexpected!r}"
            )
        return matches

    @staticmethod
    def _validate_vector_snapshot(
        records: list,
        expected_dimension: Optional[int],
        stream_name: str,
    ) -> None:
        if not records:
            return
        if expected_dimension is None or expected_dimension <= 0:
            raise ValueError(
                f"{stream_name} has records but no valid dimension."
            )
        for index, record in enumerate(records):
            try:
                vector = np.asarray(
                    record["embedding"],
                    dtype=np.float32,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{stream_name} record {index} has invalid embedding."
                ) from exc
            if vector.shape != (expected_dimension,):
                raise ValueError(
                    f"{stream_name} record {index} has dimension "
                    f"{vector.size}, expected {expected_dimension}."
                )
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"{stream_name} record {index} is non-finite."
                )
            if not np.isclose(
                np.linalg.norm(vector),
                1.0,
                atol=1e-3,
                rtol=1e-3,
            ):
                raise ValueError(
                    f"{stream_name} record {index} is not L2-normalized."
                )

    def _validate_post_index(
        self,
        snapshot: VideoSnapshot,
        *,
        visual_records: list,
        asr_emb_records: list,
        summary_emb_records: list,
        ocr_emb_records: list,
        ocr_text_records: list,
        asr_text_records: list,
        summary_text_records: list,
        metadata_records: list,
        object_records: list,
        visual_dim: Optional[int],
        text_dim: Optional[int],
        ocr_dim: Optional[int],
        video_record: Optional[dict] = None,
    ) -> None:
        """Validate counts, join IDs, vector shape, and norm after inserts."""
        mismatch_details: List[str] = []
        if not self._snapshot_matches_inputs(
            snapshot,
            visual_records=visual_records,
            asr_emb_records=asr_emb_records,
            summary_emb_records=summary_emb_records,
            ocr_emb_records=ocr_emb_records,
            ocr_text_records=ocr_text_records,
            asr_text_records=asr_text_records,
            summary_text_records=summary_text_records,
            metadata_records=metadata_records,
            object_records=object_records,
            video_record=video_record,
            mismatch_details=mismatch_details,
        ):
            raise ValueError(
                "Post-index record counts or identifiers do not match "
                "the producer artifacts: "
                + "; ".join(mismatch_details)
            )

        visual_ids = {
            record["frame_id"]
            for record in snapshot.milvus[VISUAL_COLLECTION]
        }
        metadata_ids = {
            record["frame_id"]
            for record in snapshot.metadata
        }
        if not visual_ids or visual_ids != metadata_ids:
            raise ValueError(
                "Post-index visual and metadata frame IDs do not match."
            )

        ocr_ids = {
            record["frame_id"]
            for record in snapshot.milvus[OCR_COLLECTION]
        }
        ocr_text_ids = {
            document["_source"]["frame_id"]
            for document in snapshot.elasticsearch[OCR_INDEX]
        }
        object_ids = {
            record["frame_id"]
            for record in snapshot.objects
        }
        if not (ocr_ids | ocr_text_ids | object_ids).issubset(
            metadata_ids
        ):
            raise ValueError(
                "Post-index OCR/object frame IDs are not a subset of "
                "metadata frame IDs."
            )

        asr_vector_ids = {
            (record["video_id"], str(record["interval_id"]))
            for record in snapshot.milvus[ASR_COLLECTION]
        }
        asr_text_ids = {
            (
                document["_source"]["video_id"],
                str(document["_source"]["interval_id"]),
            )
            for document in snapshot.elasticsearch[ASR_INDEX]
        }
        if asr_vector_ids != asr_text_ids:
            raise ValueError(
                "Post-index ASR vector and text interval IDs do not match."
            )

        summary_vector_ids = {
            record["video_id"]
            for record in snapshot.milvus[SUMMARY_COLLECTION]
        }
        summary_text_ids = {
            document["_source"]["video_id"]
            for document in snapshot.elasticsearch[SUMMARY_INDEX]
        }
        if summary_vector_ids != summary_text_ids:
            raise ValueError(
                "Post-index summary vector and text video IDs do not match."
            )

        self._validate_vector_snapshot(
            snapshot.milvus[VISUAL_COLLECTION],
            visual_dim,
            "visual",
        )
        self._validate_vector_snapshot(
            snapshot.milvus[ASR_COLLECTION],
            text_dim,
            "ASR",
        )
        self._validate_vector_snapshot(
            snapshot.milvus[SUMMARY_COLLECTION],
            text_dim,
            "summary",
        )
        self._validate_vector_snapshot(
            snapshot.milvus[OCR_COLLECTION],
            ocr_dim or text_dim,
            "OCR",
        )

    def _insert_batched(self, items: list, insert_fn, batch_size: int):
        """Helper to insert items in batches."""
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            insert_fn(batch)

    def _bulk_index_exact(
        self,
        index_name: str,
        documents: list,
        id_field: str,
    ) -> None:
        """Require Elasticsearch to acknowledge every document in a batch."""
        inserted_count = self.es.bulk_index(
            index_name,
            documents,
            id_field=id_field,
        )
        if inserted_count != len(documents):
            raise RuntimeError(
                f"Elasticsearch indexed {inserted_count}/"
                f"{len(documents)} documents into {index_name}."
            )

    def process_video(
        self,
        video_id: str,
        data_dir: Path,
        visual_dim: Optional[int],
        text_dim: Optional[int],
        ocr_dim: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        """
        Process a single video: load data → delete old → insert Milvus →
        insert ES → insert SQLite.

        Returns True if successful, False if failed.
        """
        logger.info(f"--- Processing video: {video_id} ---")

        # ===== PHASE 0: Load all data =====
        visual_records = load_visual_embeddings(data_dir, video_id)
        asr_emb_records = load_text_asr_embeddings(data_dir, video_id)
        summary_emb_records = load_text_summary_embeddings(data_dir, video_id)
        ocr_emb_records = load_text_ocr_embeddings(data_dir, video_id)
        ocr_text_records = load_ocr_texts(data_dir, video_id)
        asr_text_records = load_asr_transcripts(data_dir, video_id)
        summary_text_records = load_video_summary(data_dir, video_id)
        video_record = load_video_metadata(data_dir, video_id)
        metadata_records, object_records = load_metadata_and_objects(data_dir, video_id)

        core_errors = []
        if not visual_records:
            core_errors.append("visual embeddings are missing")
        if not metadata_records:
            core_errors.append("frame metadata is missing")
        if visual_records and (visual_dim is None or visual_dim <= 0):
            core_errors.append("visual embedding dimension is missing")
        if (
            (asr_emb_records or summary_emb_records)
            and (text_dim is None or text_dim <= 0)
        ):
            core_errors.append("text embedding dimension is missing")
        effective_ocr_dim = ocr_dim or text_dim
        if (
            ocr_emb_records
            and (effective_ocr_dim is None or effective_ocr_dim <= 0)
        ):
            core_errors.append("OCR embedding dimension is missing")
        if core_errors:
            logger.error(
                "Core artifact validation failed for %s: %s",
                video_id,
                "; ".join(core_errors),
            )
            return False

        try:
            # A complete snapshot is mandatory before any destructive cleanup.
            snapshot = self._capture_snapshot(video_id)
        except Exception:
            logger.exception(
                "Could not snapshot existing data for %s; replacement "
                "was aborted before delete.",
                video_id,
            )
            return False

        if not force and self._snapshot_matches_inputs(
            snapshot,
            visual_records=visual_records,
            asr_emb_records=asr_emb_records,
            summary_emb_records=summary_emb_records,
            ocr_emb_records=ocr_emb_records,
            ocr_text_records=ocr_text_records,
            asr_text_records=asr_text_records,
            summary_text_records=summary_text_records,
            metadata_records=metadata_records,
            object_records=object_records,
            video_record=video_record,
        ):
            logger.info(
                "Skipping %s because all indexed record identities match. "
                "Use --force to replace them.",
                video_id,
            )
            return True

        try:
            # ===== PHASE 1: Delete old records (idempotent cleanup) =====
            self._delete_video_from_all(video_id)

            # ===== PHASE 2: Insert into Milvus =====
            if visual_records and visual_dim:
                self._insert_batched(
                    visual_records,
                    lambda batch: self.milvus.insert_batch(VISUAL_COLLECTION, batch, visual_dim),
                    self.batch_size,
                )

            if asr_emb_records and text_dim:
                self._insert_batched(
                    asr_emb_records,
                    lambda batch: self.milvus.insert_batch(ASR_COLLECTION, batch, text_dim),
                    self.batch_size,
                )

            if summary_emb_records and text_dim:
                self._insert_batched(
                    summary_emb_records,
                    lambda batch: self.milvus.insert_batch(SUMMARY_COLLECTION, batch, text_dim),
                    self.batch_size,
                )

            # OCR embeddings — use ocr_dim if detected, fallback to text_dim
            if ocr_emb_records and effective_ocr_dim:
                self._insert_batched(
                    ocr_emb_records,
                    lambda batch: self.milvus.insert_batch(OCR_COLLECTION, batch, effective_ocr_dim),
                    self.batch_size,
                )

            # ===== PHASE 3: Insert into Elasticsearch =====
            if ocr_text_records:
                self._insert_batched(
                    ocr_text_records,
                    lambda batch: self._bulk_index_exact(
                        OCR_INDEX,
                        batch,
                        id_field="frame_id",
                    ),
                    self.batch_size,
                )

            if asr_text_records:
                # Composite ID for ASR: video_id + interval_id
                for rec in asr_text_records:
                    rec["_doc_id"] = f"{rec['video_id']}_{rec['interval_id']}"
                self._insert_batched(
                    asr_text_records,
                    lambda batch: self._bulk_index_exact(
                        ASR_INDEX,
                        batch,
                        id_field="_doc_id",
                    ),
                    self.batch_size,
                )

            if summary_text_records:
                self._insert_batched(
                    summary_text_records,
                    lambda batch: self._bulk_index_exact(
                        SUMMARY_INDEX,
                        batch,
                        id_field="video_id",
                    ),
                    self.batch_size,
                )

            # ===== PHASE 4: Insert into SQLite =====
            self.tabular.insert_video_batch([video_record])

            if metadata_records:
                self._insert_batched(
                    metadata_records,
                    self.tabular.insert_metadata_batch,
                    self.batch_size,
                )

            if object_records:
                self._insert_batched(
                    object_records,
                    self.tabular.insert_objects_batch,
                    self.batch_size,
                )

            # ===== PHASE 5: Read-after-write commit gate =====
            indexed_snapshot = self._capture_snapshot(video_id)
            self._validate_post_index(
                indexed_snapshot,
                visual_records=visual_records,
                asr_emb_records=asr_emb_records,
                summary_emb_records=summary_emb_records,
                ocr_emb_records=ocr_emb_records,
                ocr_text_records=ocr_text_records,
                asr_text_records=asr_text_records,
                summary_text_records=summary_text_records,
                metadata_records=metadata_records,
                object_records=object_records,
                visual_dim=visual_dim,
                text_dim=text_dim,
                ocr_dim=effective_ocr_dim,
                video_record=video_record,
            )

        except Exception:
            logger.exception(
                "Replacement failed for %s; restoring last-known-good "
                "snapshot.",
                video_id,
            )
            try:
                self._restore_snapshot(video_id, snapshot)
            except Exception:
                logger.critical(
                    "Snapshot restore failed for %s.",
                    video_id,
                    exc_info=True,
                )
            return False

        logger.info(f"Successfully processed video: {video_id}")
        return True

    def run(
        self,
        data_dir: Path,
        force: bool = False,
        reset_all: bool = False,
    ):
        """Connect all backends, run indexing, and always release clients."""
        attempted_clients = []
        try:
            for client in (self.milvus, self.es, self.tabular):
                attempted_clients.append(client)
                client.connect()
            return self._run_connected(
                data_dir,
                force=force,
                reset_all=reset_all,
            )
        finally:
            for client in reversed(attempted_clients):
                try:
                    client.disconnect()
                except Exception:
                    logger.exception(
                        "Failed to disconnect %s.",
                        client.__class__.__name__,
                    )

    def _run_connected(
        self,
        data_dir: Path,
        force: bool = False,
        reset_all: bool = False,
    ):
        """
        Run the full indexing pipeline for all discovered videos.

        Args:
            data_dir: Root data directory.
            force: If True, re-process all videos (delete + re-insert).
            reset_all: If True, drop all DBs and recreate schemas first.
        """
        if reset_all:
            logger.warning("Resetting all databases...")
            self.milvus.reset()
            self.es.reset()
            self.tabular.reset()

        # Create schemas
        self.es.create_indices()
        self.tabular.create_tables()

        # Detect dimensions dynamically
        visual_dim = None
        text_dim = None

        # Try visual embeddings
        for visual_dir in [data_dir / "embeddings" / "visual", data_dir / "embeddings"]:
            if visual_dir.exists():
                visual_dim = detect_embedding_dim(visual_dir)
                if visual_dim:
                    break

        # Try text embeddings (ASR as representative)
        text_asr_dir = data_dir / "embeddings" / "text_asr"
        if text_asr_dir.exists():
            text_dim = detect_embedding_dim(text_asr_dir)

        # Fallback: try text_summary
        if text_dim is None:
            text_summary_dir = data_dir / "embeddings" / "text_summary"
            if text_summary_dir.exists():
                text_dim = detect_embedding_dim(text_summary_dir)

        logger.info(f"Detected dimensions — Visual: {visual_dim}, Text: {text_dim}")

        # Try OCR text embeddings (may differ from ASR text dim)
        ocr_dim = None
        text_ocr_dir = data_dir / "embeddings" / "text_ocr"
        if text_ocr_dir.exists():
            ocr_dim = detect_embedding_dim(text_ocr_dir)
        # Fallback: assume same dim as other text embeddings
        if ocr_dim is None:
            ocr_dim = text_dim
        logger.info(f"Detected OCR embedding dim: {ocr_dim}")

        # Provision new collections or audit every existing vector contract
        # before any per-video snapshot/delete/insert transaction begins.
        if visual_dim:
            self.milvus.create_collection_if_not_exists(
                VISUAL_COLLECTION,
                visual_dim,
            )
        if text_dim:
            self.milvus.create_collection_if_not_exists(
                ASR_COLLECTION,
                text_dim,
            )
            self.milvus.create_collection_if_not_exists(
                SUMMARY_COLLECTION,
                text_dim,
            )
        if ocr_dim:
            self.milvus.create_collection_if_not_exists(
                OCR_COLLECTION,
                ocr_dim,
            )

        # Discover and process videos
        video_ids = discover_video_ids(data_dir)

        succeeded = []
        failed = []

        for video_id in tqdm(video_ids, desc="Indexing videos"):
            ok = self.process_video(
                video_id,
                data_dir,
                visual_dim,
                text_dim,
                ocr_dim,
                force=force,
            )
            if ok:
                succeeded.append(video_id)
            else:
                failed.append(video_id)

        # Summary
        logger.info(f"Indexing complete. Success: {len(succeeded)}, Failed: {len(failed)}")
        if failed:
            logger.warning(f"Failed video IDs: {failed}")
            raise RuntimeError(
                "Indexing failed for video IDs: "
                + ", ".join(failed)
            )
