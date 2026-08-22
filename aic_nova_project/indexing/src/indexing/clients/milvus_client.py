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
            collection = Collection(name, using=self.alias)
            self._audit_collection(collection, name, dim)
            logger.info(
                "Collection '%s' already exists and passed schema audit.",
                name,
            )
            return collection

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

    @staticmethod
    def _field_contract(field: FieldSchema) -> Dict[str, Any]:
        """Extract stable field properties for exact schema comparison."""
        return {
            "name": field.name,
            "dtype": field.dtype,
            "is_primary": bool(getattr(field, "is_primary", False)),
            "auto_id": bool(getattr(field, "auto_id", False)),
            "params": dict(getattr(field, "params", {}) or {}),
        }

    def _audit_collection(
        self,
        collection: Collection,
        name: str,
        dim: int,
    ) -> None:
        schema_builders = {
            VISUAL_COLLECTION: _build_visual_schema,
            ASR_COLLECTION: _build_asr_schema,
            SUMMARY_COLLECTION: _build_summary_schema,
            OCR_COLLECTION: _build_ocr_schema,
        }
        builder = schema_builders.get(name)
        if builder is None:
            raise ValueError(f"Unknown collection name: {name}")

        expected_fields = [
            self._field_contract(field)
            for field in builder(dim).fields
        ]
        actual_fields = [
            self._field_contract(field)
            for field in collection.schema.fields
        ]
        if actual_fields != expected_fields:
            raise ValueError(
                f"Milvus schema contract mismatch for '{name}'."
            )

        embedding_indexes = [
            index
            for index in collection.indexes
            if getattr(index, "field_name", None) == "embedding"
        ]
        if len(embedding_indexes) != 1:
            raise ValueError(
                f"Milvus schema contract mismatch for '{name}': "
                "expected exactly one embedding index."
            )
        actual_index = getattr(embedding_indexes[0], "params", {}) or {}
        if (
            actual_index.get("index_type")
            != HNSW_INDEX_PARAMS["index_type"]
            or actual_index.get("metric_type")
            != HNSW_INDEX_PARAMS["metric_type"]
            or actual_index.get("params")
            != HNSW_INDEX_PARAMS["params"]
        ):
            raise ValueError(
                f"Milvus index contract mismatch for '{name}'."
            )

    def insert_batch(
        self,
        collection_name: str,
        records: List[Dict[str, Any]],
        dim: int,
        *,
        flush: bool = True,
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
        if flush:
            collection.flush()
        return result.insert_count

    def flush_collections(self, collection_names: List[str]) -> None:
        """Make deferred fresh-rebuild inserts durable and searchable."""
        for name in collection_names:
            if not utility.has_collection(name, using=self.alias):
                raise ValueError(
                    f"Cannot flush missing Milvus collection '{name}'."
                )
            Collection(name, using=self.alias).flush()

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

    def snapshot_by_video_id(
        self,
        collection_name: str,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        """Read restorable records before a destructive per-video replace."""
        if not utility.has_collection(collection_name, using=self.alias):
            return []

        collection = Collection(collection_name, using=self.alias)
        collection.load()
        output_fields = [
            field.name
            for field in collection.schema.fields
            if not (
                getattr(field, "is_primary", False)
                and getattr(field, "auto_id", False)
            )
        ]
        escaped_video_id = (
            video_id.replace("\\", "\\\\").replace('"', '\\"')
        )
        records = collection.query(
            expr=f'video_id == "{escaped_video_id}"',
            output_fields=output_fields,
        )
        return [
            {
                key: value
                for key, value in record.items()
                if key != "pk"
            }
            for record in records
        ]

    def reset(self):
        """Drop all managed collections."""
        for name in [VISUAL_COLLECTION, ASR_COLLECTION, SUMMARY_COLLECTION, OCR_COLLECTION]:
            if utility.has_collection(name, using=self.alias):
                utility.drop_collection(name, using=self.alias)
                logger.info(f"Dropped collection '{name}'.")
