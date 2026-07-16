"""
Verification script: Verify frame_id consistency across all 3 databases.

Queries 1 random record from each DB (Milvus, Elasticsearch, SQLite)
and prints the frame_id to confirm they all use the same Global ID format
(e.g., V001_00000_015).

Usage:
    python verify_frame_id_consistency.py \
        --data-dir data/processed \
        --milvus-uri http://localhost:19530 \
        --es-uri http://localhost:9200 \
        --db-uri sqlite:///data/metadata.db
"""

import argparse
import sqlite3
from pathlib import Path

from pymilvus import connections, Collection, utility
from elasticsearch import Elasticsearch


def verify_milvus(uri: str):
    """Query 1 random record from each Milvus collection and print frame_id."""
    print("\n" + "=" * 60)
    print("MILVUS — Checking frame_id format in all collections")
    print("=" * 60)

    connections.connect(alias="verify", uri=uri)

    for coll_name in ["visual_features", "asr_features", "summary_features", "ocr_features"]:
        if not utility.has_collection(coll_name, using="verify"):
            print(f"  [{coll_name}] Collection does not exist. SKIP.")
            continue

        collection = Collection(coll_name, using="verify")
        collection.load()

        # Query 1 record
        if coll_name in ("visual_features", "ocr_features"):
            results = collection.query(
                expr="pk >= 0",
                output_fields=["frame_id", "video_id"],
                limit=1,
            )
        elif coll_name == "asr_features":
            results = collection.query(
                expr="pk >= 0",
                output_fields=["video_id", "interval_id"],
                limit=1,
            )
        else:  # summary_features
            results = collection.query(
                expr="pk >= 0",
                output_fields=["video_id"],
                limit=1,
            )

        if results:
            rec = results[0]
            frame_id = rec.get("frame_id", "N/A")
            video_id = rec.get("video_id", "N/A")
            interval_id = rec.get("interval_id", "N/A")
            if coll_name in ("visual_features", "ocr_features"):
                print(f"  [{coll_name}] frame_id = {frame_id}, video_id = {video_id}")
            elif coll_name == "asr_features":
                print(f"  [{coll_name}] video_id = {video_id}, interval_id = {interval_id}")
            else:
                print(f"  [{coll_name}] video_id = {video_id}")
        else:
            print(f"  [{coll_name}] EMPTY — no records.")

    connections.disconnect("verify")


def verify_elasticsearch(uri: str):
    """Query 1 random record from each Elasticsearch index and print frame_id."""
    print("\n" + "=" * 60)
    print("ELASTICSEARCH — Checking frame_id format in all indices")
    print("=" * 60)

    es = Elasticsearch(uri)

    for index_name in ["ocr_texts", "asr_transcripts", "video_summaries"]:
        if not es.indices.exists(index=index_name):
            print(f"  [{index_name}] Index does not exist. SKIP.")
            continue

        resp = es.search(index=index_name, body={"size": 1, "query": {"match_all": {}}})
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            src = hits[0]["_source"]
            doc_id = hits[0]["_id"]
            frame_id = src.get("frame_id", "N/A")
            video_id = src.get("video_id", "N/A")

            if index_name == "ocr_texts":
                print(f"  [{index_name}] _id = {doc_id}, frame_id = {frame_id}, video_id = {video_id}")
            elif index_name == "asr_transcripts":
                interval_id = src.get("interval_id", "N/A")
                print(f"  [{index_name}] _id = {doc_id}, video_id = {video_id}, interval_id = {interval_id}")
            else:
                print(f"  [{index_name}] _id = {doc_id}, video_id = {video_id}")
        else:
            print(f"  [{index_name}] EMPTY — no records.")

    es.close()


def verify_sqlite(db_uri: str):
    """Query 1 random record from each SQLite table and print frame_id."""
    print("\n" + "=" * 60)
    print("SQLITE — Checking frame_id format in tables")
    print("=" * 60)

    if db_uri.startswith("sqlite:///"):
        db_uri = db_uri[len("sqlite:///"):]

    db_path = Path(db_uri)
    if not db_path.exists():
        print(f"  Database file not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # metadata
    cursor = conn.execute("SELECT frame_id, video_id, shot_id, timestamp FROM metadata LIMIT 1")
    row = cursor.fetchone()
    if row:
        print(f"  [metadata] frame_id = {row['frame_id']}, video_id = {row['video_id']}, "
              f"shot_id = {row['shot_id']}, timestamp = {row['timestamp']}")
    else:
        print("  [metadata] EMPTY — no records.")

    # objects
    cursor = conn.execute("SELECT frame_id, label, confidence FROM objects LIMIT 1")
    row = cursor.fetchone()
    if row:
        print(f"  [objects]  frame_id = {row['frame_id']}, label = {row['label']}, "
              f"confidence = {row['confidence']}")
    else:
        print("  [objects]  EMPTY — no records.")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Verify frame_id consistency across Milvus, ES, and SQLite"
    )
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--es-uri", default="http://localhost:9200")
    parser.add_argument("--db-uri", default="sqlite:///data/metadata.db")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  FRAME_ID CONSISTENCY VERIFICATION ACROSS 3 DATABASES  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        verify_milvus(args.milvus_uri)
    except Exception as e:
        print(f"\n  Milvus verification FAILED: {e}")

    try:
        verify_elasticsearch(args.es_uri)
    except Exception as e:
        print(f"\n  Elasticsearch verification FAILED: {e}")

    try:
        verify_sqlite(args.db_uri)
    except Exception as e:
        print(f"\n  SQLite verification FAILED: {e}")

    print("\n" + "=" * 60)
    print("DONE. Compare frame_id values above — they should all")
    print("follow the Global ID format: {video_id}_{shot_id}_{position}")
    print("Example: V001_00000_015")
    print("=" * 60)


if __name__ == "__main__":
    main()
