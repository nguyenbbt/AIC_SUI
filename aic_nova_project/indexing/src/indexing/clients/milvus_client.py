"""
Milvus Vector Database client for indexing dense embeddings.

Manages 4 collections:
- visual_features: frame-level visual embeddings
- asr_features: ASR interval-level text embeddings
- summary_features: video-level summary embeddings
- ocr_features: OCR text embeddings per keyframe

Vector dimensions are detected dynamically at runtime from the data,
NOT hard-coded.
"""

import logging
from typing import List, Dict, Any, Optional

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

logger = logging.getLogger(__name__)

# Collection names
VISUAL_COLLECTION = "visual_features"
ASR_COLLECTION = "asr_features"
SUMMARY_COLLECTION = "summary_features"
OCR_COLLECTION = "ocr_features"

# HNSW index params — Inner Product (IP) metric is equivalent to Cosine
# when vectors are L2-normalized (which they are from Module 2 & 6).
HNSW_INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256},
}

SEARCH_PARAMS = {"metric_type": "IP", "params": {"ef": 128}}


def _build_visual_schema(dim: int) -> CollectionSchema:
    """Schema for visual_features collection."""
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="frame_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="shot_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(fields, description="Frame-level visual embeddings")


def _build_asr_schema(dim: int) -> CollectionSchema:
    """Schema for asr_features collection."""
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="interval_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="start_time_sec", dtype=DataType.FLOAT),
        FieldSchema(name="end_time_sec", dtype=DataType.FLOAT),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(fields, description="ASR interval text embeddings")


def _build_summary_schema(dim: int) -> CollectionSchema:
    """Schema for summary_features collection."""
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(fields, description="Video-level summary embeddings")


def _build_ocr_schema(dim: int) -> CollectionSchema:
    """Schema for ocr_features collection."""
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="frame_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(fields, description="OCR text embeddings per keyframe")


class MilvusVectorClient:
    """Client for managing Milvus vector collections."""

    def __init__(self, uri: str = "http://localhost:19530", alias: str = "default"):
        self.uri = uri
        self.alias = alias
        self._connected = False

    def connect(self):
        """Establish connection to Milvus server."""
        if self._connected:
            return
        logger.info(f"Connecting to Milvus at {self.uri}...")
        connections.connect(alias=self.alias, uri=self.uri)
        self._connected = True
        logger.info("Milvus connected.")

    def disconnect(self):
        """Disconnect from Milvus server."""
        if self._connected:
            connections.disconnect(alias=self.alias)
            self._connected = False

    def create_collection_if_not_exists(
        self, name: str, dim: int
    ) -> Collection:
        """
        Creates a collection with the given name and vector dimension
        if it does not already exist. Dimension is a runtime parameter.
        """
        if utility.has_collection(name, using=self.alias):
            logger.info(f"Collection '{name}' already exists.")
            return Collection(name, using=self.alias)

        schema_builders = {
            VISUAL_COLLECTION: _build_visual_schema,
            ASR_COLLECTION: _build_asr_schema,
            SUMMARY_COLLECTION: _build_summary_schema,
            OCR_COLLECTION: _build_ocr_schema,
        }

        builder = schema_builders.get(name)
        if builder is None:
            raise ValueError(f"Unknown collection name: {name}")

        schema = builder(dim)
        collection = Collection(name=name, schema=schema, using=self.alias)

        # Create HNSW index on the embedding field
        collection.create_index(field_name="embedding", index_params=HNSW_INDEX_PARAMS)
        logger.info(f"Created collection '{name}' with dim={dim} and HNSW index.")

        return collection

    def insert_batch(
        self, collection_name: str, records: List[Dict[str, Any]], dim: int
    ) -> int:
        """
        Inserts a batch of records into the specified collection.
        Returns the number of inserted entities.
        """
        if not records:
            return 0

        collection = self.create_collection_if_not_exists(collection_name, dim)

        # pymilvus insert expects list-of-columns format
        columns: Dict[str, list] = {}
        for key in records[0]:
            if key == "pk":
                continue  # auto_id
            columns[key] = [r[key] for r in records]

        result = collection.insert(list(columns.values()), fields=list(columns.keys()))
        collection.flush()
        return result.insert_count

    def delete_by_video_id(self, collection_name: str, video_id: str):
        """Delete all entities for a given video_id from a collection."""
        if not utility.has_collection(collection_name, using=self.alias):
            return

        collection = Collection(collection_name, using=self.alias)
        collection.load()
        expr = f'video_id == "{video_id}"'
        collection.delete(expr)
        collection.flush()
        logger.info(
            f"Deleted records for video_id='{video_id}' from '{collection_name}'."
        )

    def reset(self):
        """Drop all managed collections."""
        for name in [VISUAL_COLLECTION, ASR_COLLECTION, SUMMARY_COLLECTION, OCR_COLLECTION]:
            if utility.has_collection(name, using=self.alias):
                utility.drop_collection(name, using=self.alias)
                logger.info(f"Dropped collection '{name}'.")
