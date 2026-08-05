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
from elasticsearch.helpers import bulk, scan

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
        "start_time_sec": {"type": "float"},
        "end_time_sec": {"type": "float"},
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
            self._audit_index(index_name, mapping)
            logger.info(
                "Index '%s' already exists and passed schema audit.",
                index_name,
            )
            return

        body = {
            "settings": VIETNAMESE_ANALYSIS_SETTINGS,
            "mappings": mapping,
        }
        self.client.indices.create(index=index_name, body=body)
        logger.info(f"Created index '{index_name}' with Vietnamese analyzer.")

    @staticmethod
    def _contains_contract(
        actual: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> bool:
        """Return whether actual recursively contains the expected contract."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            actual_value = actual[key]
            if isinstance(expected_value, dict):
                if not isinstance(actual_value, dict):
                    return False
                if not ESClient._contains_contract(
                    actual_value,
                    expected_value,
                ):
                    return False
            elif actual_value != expected_value:
                return False
        return True

    def _audit_index(
        self,
        index_name: str,
        mapping: Dict[str, Any],
    ) -> None:
        mapping_response = self.client.indices.get_mapping(
            index=index_name
        )
        actual_mapping = mapping_response.get(index_name, {}).get(
            "mappings",
            {},
        )
        if not self._contains_contract(actual_mapping, mapping):
            raise ValueError(
                f"Elasticsearch mapping contract mismatch for "
                f"'{index_name}'."
            )

        settings_response = self.client.indices.get_settings(
            index=index_name
        )
        index_settings = settings_response.get(index_name, {}).get(
            "settings",
            {},
        )
        actual_analysis = index_settings.get("index", {}).get(
            "analysis",
            index_settings.get("analysis", {}),
        )
        expected_analysis = VIETNAMESE_ANALYSIS_SETTINGS["analysis"]
        if not self._contains_contract(
            actual_analysis,
            expected_analysis,
        ):
            raise ValueError(
                f"Elasticsearch analyzer contract mismatch for "
                f"'{index_name}'."
            )

        try:
            self.client.indices.analyze(
                index=index_name,
                body={
                    "analyzer": "vietnamese_analyzer",
                    "text": "người",
                },
            )
        except Exception as exc:
            raise ValueError(
                f"Elasticsearch analyzer/plugin audit failed for "
                f"'{index_name}': {exc}"
            ) from exc

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

        success_count, errors = bulk(
            self.client,
            actions,
            raise_on_error=False,
            refresh="wait_for",
        )
        if errors:
            raise RuntimeError(
                f"Bulk index to '{index_name}' had {len(errors)} error(s); "
                f"first error: {errors[0]}"
            )
        if success_count != len(documents):
            raise RuntimeError(
                f"Bulk index to '{index_name}' indexed "
                f"{success_count}/{len(documents)} documents."
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

    def snapshot_by_video_id(
        self,
        index_name: str,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        """Read all documents needed to restore a failed replacement."""
        if not self.client.indices.exists(index=index_name):
            return []

        hits = scan(
            self.client,
            index=index_name,
            query={"query": {"term": {"video_id": video_id}}},
        )
        return [
            {
                "_id": hit["_id"],
                "_source": hit["_source"],
            }
            for hit in hits
        ]

    def restore_snapshot(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
    ) -> None:
        """Restore documents with their original Elasticsearch IDs."""
        if not documents:
            return

        actions = [
            {
                "_index": index_name,
                "_id": document["_id"],
                "_source": document["_source"],
            }
            for document in documents
        ]
        bulk(self.client, actions, raise_on_error=True)
        self.client.indices.refresh(index=index_name)

    def reset(self):
        """Delete all managed indices."""
        for name in [OCR_INDEX, ASR_INDEX, SUMMARY_INDEX]:
            if self.client.indices.exists(index=name):
                self.client.indices.delete(index=name)
                logger.info(f"Deleted index '{name}'.")
