"""
CLI entry point for the Multi-DB Indexing module.
"""

import argparse
import logging
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


def main():
    parser = argparse.ArgumentParser(
        description="AI Challenge 2026 - Multi-DB Indexing & Ingestion"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory (e.g., data/processed)",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default="http://localhost:19530",
        help="Milvus connection URI",
    )
    parser.add_argument(
        "--es-uri",
        type=str,
        default="http://localhost:9200",
        help="Elasticsearch connection URI",
    )
    parser.add_argument(
        "--db-uri",
        type=str,
        default="sqlite:///data/metadata.db",
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
    )


if __name__ == "__main__":
    main()
