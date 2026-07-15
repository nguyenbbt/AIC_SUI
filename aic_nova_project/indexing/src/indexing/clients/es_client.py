"""
Elasticsearch client for full-text search indexing.

Manages 3 indices:
- ocr_texts: OCR text from keyframes
- asr_transcripts: ASR cleaned text from intervals
- video_summaries: Video-level summaries

Uses a custom Vietnamese analyzer with ICU tokenizer + ICU folding
for accent-insensitive search (e.g., "nguoi" matches "người").
"""

import logging
from typing import List, Dict, Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)

# Index names
OCR_INDEX = "ocr_texts"
ASR_INDEX = "asr_transcripts"
SUMMARY_INDEX = "video_summaries"

# Shared analyzer settings for all indices
VIETNAMESE_ANALYSIS_SETTINGS = {
    "analysis": {
        "analyzer": {
            "vietnamese_analyzer": {
                "type": "custom",
                "tokenizer": "icu_tokenizer",
                "filter": ["icu_folding", "lowercase"],
            }
        }
    }
}

# Mappings per index
OCR_MAPPING = {
    "properties": {
        "frame_id": {"type": "keyword"},
        "video_id": {"type": "keyword"},
        "shot_id": {"type": "keyword"},
        "ocr_text_concat": {
            "type": "text",
            "analyzer": "vietnamese_analyzer",
        },
    }
}

ASR_MAPPING = {
    "properties": {
        "interval_id": {"type": "keyword"},
        "video_id": {"type": "keyword"},
        "start_time": {"type": "float"},
        "end_time": {"type": "float"},
        "cleaned_text": {
            "type": "text",
            "analyzer": "vietnamese_analyzer",
        },
    }
}

SUMMARY_MAPPING = {
    "properties": {
        "video_id": {"type": "keyword"},
        "summary": {
            "type": "text",
            "analyzer": "vietnamese_analyzer",
        },
    }
}


class ESClient:
    """Client for managing Elasticsearch indices for Vietnamese full-text search."""

    def __init__(self, uri: str = "http://localhost:9200"):
        self.uri = uri
        self.client: Elasticsearch | None = None

    def connect(self):
        """Establish connection to Elasticsearch."""
        logger.info(f"Connecting to Elasticsearch at {self.uri}...")
        self.client = Elasticsearch(self.uri)
        info = self.client.info()
        logger.info(f"Elasticsearch connected. Version: {info['version']['number']}")

    def disconnect(self):
        """Close Elasticsearch connection."""
        if self.client:
            self.client.close()
            self.client = None

    def _create_index_if_not_exists(
        self, index_name: str, mapping: Dict[str, Any]
    ):
        """Creates an index with Vietnamese analyzer settings if it does not exist."""
        if self.client.indices.exists(index=index_name):
            logger.info(f"Index '{index_name}' already exists.")
            return

        body = {
            "settings": VIETNAMESE_ANALYSIS_SETTINGS,
            "mappings": mapping,
        }
        self.client.indices.create(index=index_name, body=body)
        logger.info(f"Created index '{index_name}' with Vietnamese analyzer.")

    def create_indices(self):
        """Create all managed indices."""
        self._create_index_if_not_exists(OCR_INDEX, OCR_MAPPING)
        self._create_index_if_not_exists(ASR_INDEX, ASR_MAPPING)
        self._create_index_if_not_exists(SUMMARY_INDEX, SUMMARY_MAPPING)

    def bulk_index(
        self, index_name: str, documents: List[Dict[str, Any]], id_field: str
    ) -> int:
        """
        Bulk-indexes documents into the specified index.
        Uses the value of `id_field` from each document as the Elasticsearch `_id`,
        ensuring natural upsert (insert-or-replace) on re-run.

        Returns the count of successfully indexed documents.
        """
        if not documents:
            return 0

        actions = []
        for doc in documents:
            doc_id = doc.get(id_field, None)
            action = {
                "_index": index_name,
                "_source": doc,
            }
            if doc_id is not None:
                action["_id"] = str(doc_id)
            actions.append(action)

        success_count, errors = bulk(self.client, actions, raise_on_error=False)
        if errors:
            logger.warning(
                f"Bulk index to '{index_name}' had {len(errors)} error(s). "
                f"First error: {errors[0]}"
            )
        return success_count

    def delete_by_video_id(self, index_name: str, video_id: str):
        """Delete all documents matching video_id from an index."""
        if not self.client.indices.exists(index=index_name):
            return

        self.client.delete_by_query(
            index=index_name,
            body={"query": {"term": {"video_id": video_id}}},
            refresh=True,
        )
        logger.info(
            f"Deleted records for video_id='{video_id}' from index '{index_name}'."
        )

    def reset(self):
        """Delete all managed indices."""
        for name in [OCR_INDEX, ASR_INDEX, SUMMARY_INDEX]:
            if self.client.indices.exists(index=name):
                self.client.indices.delete(index=name)
                logger.info(f"Deleted index '{name}'.")
