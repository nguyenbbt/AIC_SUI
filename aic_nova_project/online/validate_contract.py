"""CLI for the read-only Offline contract validator."""

from __future__ import annotations

import argparse

from online.adapters.contract_validator import OfflineContractValidator, ValidationStatus
from online.adapters.elasticsearch import ElasticsearchSearchAdapter
from online.adapters.milvus import MilvusSearchAdapter
from online.adapters.sqlite import SQLiteReadAdapter
from online.config import OnlineDataConfig
from online.lifecycle import InfrastructureLifecycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Offline databases for Online use (read-only)")
    parser.add_argument(
        "--fail-on-partial",
        action="store_true",
        help="Return a non-zero status for PARTIAL as well as FAIL",
    )
    args = parser.parse_args()

    config = OnlineDataConfig.from_env()
    milvus = MilvusSearchAdapter(config.milvus)
    elasticsearch = ElasticsearchSearchAdapter(config.elasticsearch)
    sqlite = SQLiteReadAdapter(config.sqlite)
    lifecycle = InfrastructureLifecycle()
    lifecycle.register("milvus", milvus, required=True)
    lifecycle.register("elasticsearch", elasticsearch, required=False)
    lifecycle.register("sqlite", sqlite, required=True)

    try:
        lifecycle.start()
        report = OfflineContractValidator(
            config,
            milvus=milvus,
            elasticsearch=elasticsearch,
            sqlite=sqlite,
        ).validate()
        print(report.model_dump_json(indent=2))
        if report.status is ValidationStatus.FAIL:
            return 2
        if report.status is ValidationStatus.PARTIAL and args.fail_on_partial:
            return 1
        return 0
    finally:
        lifecycle.close()


if __name__ == "__main__":
    raise SystemExit(main())
