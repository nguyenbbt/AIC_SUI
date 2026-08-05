"""Verify join-key consistency across Milvus, Elasticsearch, and SQLite."""

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from pymilvus import Collection, connections, utility


CANONICAL_FRAME_ID = re.compile(r"^.+_[0-9]{5}_[0-9]{3}$")


@dataclass(frozen=True)
class VerificationSnapshot:
    """Join keys read from every online storage branch."""

    visual_frame_ids: Set[str]
    ocr_vector_frame_ids: Set[str]
    ocr_text_frame_ids: Set[str]
    metadata_frame_ids: Set[str]
    object_frame_ids: Set[str]
    asr_vector_ids: Set[Tuple[str, str]]
    asr_text_ids: Set[Tuple[str, str]]
    summary_vector_ids: Set[str]
    summary_text_ids: Set[str]


def _describe_difference(
    left_name: str,
    left: set,
    right_name: str,
    right: set,
) -> str:
    left_only = sorted(left - right)
    right_only = sorted(right - left)
    return (
        f"{left_name} and {right_name} do not JOIN: "
        f"{left_name}-only={left_only[:10]}, "
        f"{right_name}-only={right_only[:10]}"
    )


def build_consistency_report(
    snapshot: VerificationSnapshot,
) -> List[str]:
    """Return cross-database contract violations for matching record keys."""
    errors: List[str] = []
    frame_sets = {
        "Milvus visual": snapshot.visual_frame_ids,
        "Milvus OCR": snapshot.ocr_vector_frame_ids,
        "Elasticsearch OCR": snapshot.ocr_text_frame_ids,
        "SQLite metadata": snapshot.metadata_frame_ids,
        "SQLite objects": snapshot.object_frame_ids,
    }
    for source_name, frame_ids in frame_sets.items():
        invalid = sorted(
            frame_id
            for frame_id in frame_ids
            if not CANONICAL_FRAME_ID.fullmatch(frame_id)
        )
        if invalid:
            errors.append(
                f"{source_name} contains invalid frame IDs: {invalid[:10]}"
            )

    if snapshot.visual_frame_ids != snapshot.metadata_frame_ids:
        errors.append(
            _describe_difference(
                "Milvus visual",
                snapshot.visual_frame_ids,
                "SQLite metadata",
                snapshot.metadata_frame_ids,
            )
        )
    if snapshot.ocr_vector_frame_ids != snapshot.ocr_text_frame_ids:
        errors.append(
            _describe_difference(
                "Milvus OCR",
                snapshot.ocr_vector_frame_ids,
                "Elasticsearch OCR",
                snapshot.ocr_text_frame_ids,
            )
        )

    for source_name, frame_ids in (
        ("Milvus OCR", snapshot.ocr_vector_frame_ids),
        ("Elasticsearch OCR", snapshot.ocr_text_frame_ids),
        ("SQLite objects", snapshot.object_frame_ids),
    ):
        orphan_ids = sorted(frame_ids - snapshot.metadata_frame_ids)
        if orphan_ids:
            errors.append(
                f"{source_name} has frame IDs absent from SQLite metadata: "
                f"{orphan_ids[:10]}"
            )

    if snapshot.asr_vector_ids != snapshot.asr_text_ids:
        errors.append(
            _describe_difference(
                "Milvus ASR",
                snapshot.asr_vector_ids,
                "Elasticsearch ASR",
                snapshot.asr_text_ids,
            )
        )
    if snapshot.summary_vector_ids != snapshot.summary_text_ids:
        errors.append(
            _describe_difference(
                "Milvus summary",
                snapshot.summary_vector_ids,
                "Elasticsearch summary",
                snapshot.summary_text_ids,
            )
        )
    return errors


def _query_milvus(
    collection_name: str,
    output_fields: List[str],
) -> List[Dict]:
    if not utility.has_collection(collection_name, using="verify"):
        return []

    collection = Collection(collection_name, using="verify")
    collection.load()
    iterator = collection.query_iterator(
        batch_size=1_000,
        expr="pk >= 0",
        output_fields=output_fields,
    )
    records: List[Dict] = []
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            records.extend(batch)
    finally:
        iterator.close()
    return records


def collect_milvus_keys(uri: str) -> Dict[str, set]:
    """Read all relevant join keys from Milvus."""
    connections.connect(alias="verify", uri=uri)
    try:
        visual = _query_milvus(
            "visual_features",
            ["frame_id", "video_id"],
        )
        ocr = _query_milvus(
            "ocr_features",
            ["frame_id", "video_id"],
        )
        asr = _query_milvus(
            "asr_features",
            ["video_id", "interval_id"],
        )
        summaries = _query_milvus(
            "summary_features",
            ["video_id"],
        )
        return {
            "visual": {str(record["frame_id"]) for record in visual},
            "ocr": {str(record["frame_id"]) for record in ocr},
            "asr": {
                (str(record["video_id"]), str(record["interval_id"]))
                for record in asr
            },
            "summary": {
                str(record["video_id"])
                for record in summaries
            },
        }
    finally:
        connections.disconnect("verify")


def _scan_es_index(
    client: Elasticsearch,
    index_name: str,
) -> Iterable[Dict]:
    if not client.indices.exists(index=index_name):
        return []
    return scan(
        client,
        index=index_name,
        query={"query": {"match_all": {}}},
    )


def collect_elasticsearch_keys(uri: str) -> Dict[str, set]:
    """Read all relevant join keys from Elasticsearch."""
    client = Elasticsearch(uri)
    try:
        ocr = list(_scan_es_index(client, "ocr_texts"))
        asr = list(_scan_es_index(client, "asr_transcripts"))
        summaries = list(_scan_es_index(client, "video_summaries"))
        return {
            "ocr": {
                str(hit["_source"]["frame_id"])
                for hit in ocr
            },
            "asr": {
                (
                    str(hit["_source"]["video_id"]),
                    str(hit["_source"]["interval_id"]),
                )
                for hit in asr
            },
            "summary": {
                str(hit["_source"]["video_id"])
                for hit in summaries
            },
        }
    finally:
        client.close()


def collect_sqlite_keys(db_uri: str) -> Dict[str, set]:
    """Read all relevant join keys from SQLite."""
    if db_uri.startswith("sqlite:///"):
        db_uri = db_uri[len("sqlite:///"):]
    db_path = Path(db_uri)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    connection = sqlite3.connect(str(db_path))
    try:
        metadata = {
            str(row[0])
            for row in connection.execute(
                "SELECT frame_id FROM metadata"
            ).fetchall()
        }
        objects = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT frame_id FROM objects"
            ).fetchall()
        }
        return {"metadata": metadata, "objects": objects}
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JOIN all record IDs across Milvus, ES, and SQLite"
    )
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--es-uri", default="http://localhost:9200")
    parser.add_argument(
        "--db-uri",
        default="sqlite:///data/metadata.db",
    )
    args = parser.parse_args()

    try:
        milvus = collect_milvus_keys(args.milvus_uri)
        elasticsearch = collect_elasticsearch_keys(args.es_uri)
        sqlite = collect_sqlite_keys(args.db_uri)
    except Exception as exc:
        print(f"Verification failed while reading databases: {exc}")
        return 1

    snapshot = VerificationSnapshot(
        visual_frame_ids=milvus["visual"],
        ocr_vector_frame_ids=milvus["ocr"],
        ocr_text_frame_ids=elasticsearch["ocr"],
        metadata_frame_ids=sqlite["metadata"],
        object_frame_ids=sqlite["objects"],
        asr_vector_ids=milvus["asr"],
        asr_text_ids=elasticsearch["asr"],
        summary_vector_ids=milvus["summary"],
        summary_text_ids=elasticsearch["summary"],
    )
    errors = build_consistency_report(snapshot)
    if errors:
        print("Cross-database consistency verification FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Cross-database consistency verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
