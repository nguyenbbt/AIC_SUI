"""
CLI entry point for the Multi-DB Indexing module.
"""

import argparse
import logging
import os
from pathlib import Path

from src.indexing.clients.milvus_client import MilvusVectorClient
from src.indexing.clients.es_client import ESClient
from src.indexing.clients.tabular_client import TabularClient
from src.indexing.orchestrator import IndexingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, including m-gpux-rewritable env defaults."""
    parser = argparse.ArgumentParser(
        description="AI Challenge 2026 - Multi-DB Indexing & Ingestion"
    )
    data_dir_default = os.getenv("DATA_DIR")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(Path(data_dir_default) if data_dir_default else None),
        required=data_dir_default is None,
        help="Root data directory (e.g., data/processed)",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=os.getenv("MILVUS_URI", "http://localhost:19530"),
        help="Milvus connection URI",
    )
    parser.add_argument(
        "--es-uri",
        type=str,
        default=os.getenv("ES_URI", "http://localhost:9200"),
        help="Elasticsearch connection URI",
    )
    parser.add_argument(
        "--db-uri",
        type=str,
        default=os.getenv("DB_URI", "sqlite:///data/metadata.db"),
        help="SQLite database URI (default: sqlite:///data/metadata.db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for insert operations",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Drop and recreate all DB schemas before indexing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of all videos",
    )
    parser.add_argument(
        "--bulk-rebuild",
        action="store_true",
        help=(
            "Defer Milvus flush and Elasticsearch refresh during a fresh "
            "full rebuild; requires --reset-all"
        ),
    )
    parser.add_argument(
        "--video-id",
        dest="video_ids",
        action="append",
        help=(
            "Process only this discovered video ID. Repeat the option to "
            "repair multiple videos without scanning/replacing the corpus."
        ),
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Flush all Milvus collections and refresh all Elasticsearch "
            "indices after a successful run."
        ),
    )
    parser.add_argument(
        "--unpublished-repair",
        action="store_true",
        help=(
            "Repair selected video IDs in an unpublished candidate without "
            "capturing per-video snapshots; requires --video-id and "
            "--finalize, and must be followed by full contract validation."
        ),
    )

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    milvus_client = MilvusVectorClient(uri=args.milvus_uri)
    es_client = ESClient(uri=args.es_uri)
    tabular_client = TabularClient(db_uri=args.db_uri)

    orchestrator = IndexingOrchestrator(
        milvus_client=milvus_client,
        es_client=es_client,
        tabular_client=tabular_client,
        batch_size=args.batch_size,
    )

    orchestrator.run(
        data_dir=args.data_dir,
        force=args.force,
        reset_all=args.reset_all,
        bulk_rebuild=args.bulk_rebuild,
        video_ids=args.video_ids,
        finalize=args.finalize,
        unpublished_repair=args.unpublished_repair,
    )


if __name__ == "__main__":
    main()
