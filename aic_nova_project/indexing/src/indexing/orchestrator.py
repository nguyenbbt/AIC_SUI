"""
Orchestrator: coordinates data ingestion into Milvus, Elasticsearch, and SQLite.

Implements per-video transactional semantics:
- Delete-then-Insert (idempotent upsert)
- Rollback on partial failure (if ES fails after Milvus succeeds,
  Milvus records for that video are rolled back)
- Graceful degradation (missing OCR/Object data is silently skipped)
"""

import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .clients.milvus_client import (
    MilvusVectorClient,
    VISUAL_COLLECTION,
    ASR_COLLECTION,
    SUMMARY_COLLECTION,
)
from .clients.es_client import ESClient, OCR_INDEX, ASR_INDEX, SUMMARY_INDEX
from .clients.tabular_client import TabularClient
from .data_loader import (
    discover_video_ids,
    detect_embedding_dim,
    load_visual_embeddings,
    load_text_asr_embeddings,
    load_text_summary_embeddings,
    load_ocr_texts,
    load_asr_transcripts,
    load_video_summary,
    load_metadata_and_objects,
)

logger = logging.getLogger(__name__)


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

        # Elasticsearch
        self.es.delete_by_video_id(OCR_INDEX, video_id)
        self.es.delete_by_video_id(ASR_INDEX, video_id)
        self.es.delete_by_video_id(SUMMARY_INDEX, video_id)

        # SQLite
        self.tabular.delete_by_video_id(video_id)

    def _rollback_milvus(self, video_id: str):
        """Rollback Milvus records for a video."""
        try:
            self.milvus.delete_by_video_id(VISUAL_COLLECTION, video_id)
            self.milvus.delete_by_video_id(ASR_COLLECTION, video_id)
            self.milvus.delete_by_video_id(SUMMARY_COLLECTION, video_id)
        except Exception as e:
            logger.error(f"Failed to rollback Milvus for {video_id}: {e}")

    def _rollback_es(self, video_id: str):
        """Rollback Elasticsearch records for a video."""
        try:
            self.es.delete_by_video_id(OCR_INDEX, video_id)
            self.es.delete_by_video_id(ASR_INDEX, video_id)
            self.es.delete_by_video_id(SUMMARY_INDEX, video_id)
        except Exception as e:
            logger.error(f"Failed to rollback ES for {video_id}: {e}")

    def _insert_batched(self, items: list, insert_fn, batch_size: int):
        """Helper to insert items in batches."""
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            insert_fn(batch)

    def process_video(
        self,
        video_id: str,
        data_dir: Path,
        visual_dim: Optional[int],
        text_dim: Optional[int],
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
        ocr_text_records = load_ocr_texts(data_dir, video_id)
        asr_text_records = load_asr_transcripts(data_dir, video_id)
        summary_text_records = load_video_summary(data_dir, video_id)
        metadata_records, object_records = load_metadata_and_objects(data_dir, video_id)

        # ===== PHASE 1: Delete old records (idempotent cleanup) =====
        self._delete_video_from_all(video_id)

        # ===== PHASE 2: Insert into Milvus =====
        try:
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

        except Exception as e:
            logger.error(f"Milvus insert failed for {video_id}: {e}")
            self._rollback_milvus(video_id)
            return False

        # ===== PHASE 3: Insert into Elasticsearch =====
        try:
            if ocr_text_records:
                self._insert_batched(
                    ocr_text_records,
                    lambda batch: self.es.bulk_index(OCR_INDEX, batch, id_field="frame_id"),
                    self.batch_size,
                )

            if asr_text_records:
                # Composite ID for ASR: video_id + interval_id
                for rec in asr_text_records:
                    rec["_doc_id"] = f"{rec['video_id']}_{rec['interval_id']}"
                self._insert_batched(
                    asr_text_records,
                    lambda batch: self.es.bulk_index(ASR_INDEX, batch, id_field="_doc_id"),
                    self.batch_size,
                )

            if summary_text_records:
                self._insert_batched(
                    summary_text_records,
                    lambda batch: self.es.bulk_index(SUMMARY_INDEX, batch, id_field="video_id"),
                    self.batch_size,
                )

        except Exception as e:
            logger.error(f"Elasticsearch insert failed for {video_id}: {e}. Rolling back Milvus.")
            self._rollback_milvus(video_id)
            return False

        # ===== PHASE 4: Insert into SQLite =====
        try:
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

        except Exception as e:
            logger.error(
                f"SQLite insert failed for {video_id}: {e}. Rolling back Milvus + ES."
            )
            self._rollback_milvus(video_id)
            self._rollback_es(video_id)
            return False

        logger.info(f"Successfully processed video: {video_id}")
        return True

    def run(self, data_dir: Path, force: bool = False, reset_all: bool = False):
        """
        Run the full indexing pipeline for all discovered videos.

        Args:
            data_dir: Root data directory.
            force: If True, re-process all videos (delete + re-insert).
            reset_all: If True, drop all DBs and recreate schemas first.
        """
        # Connect all clients
        self.milvus.connect()
        self.es.connect()
        self.tabular.connect()

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

        # Discover and process videos
        video_ids = discover_video_ids(data_dir)

        succeeded = []
        failed = []

        for video_id in tqdm(video_ids, desc="Indexing videos"):
            ok = self.process_video(video_id, data_dir, visual_dim, text_dim)
            if ok:
                succeeded.append(video_id)
            else:
                failed.append(video_id)

        # Summary
        logger.info(f"Indexing complete. Success: {len(succeeded)}, Failed: {len(failed)}")
        if failed:
            logger.warning(f"Failed video IDs: {failed}")
