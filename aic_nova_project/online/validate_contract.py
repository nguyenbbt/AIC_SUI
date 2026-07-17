"""CLI for the read-only Offline contract validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--encoder-smoke-json",
        type=Path,
        help=(
            "Optional JSON object mapping collection names to a single "
            "precomputed normalized vector; no model is loaded by the CLI"
        ),
    )
    args = parser.parse_args()

    try:
        config = OnlineDataConfig.from_env()
        smoke_vectors = _load_smoke_vectors(args.encoder_smoke_json)
        milvus = MilvusSearchAdapter(config.milvus)
        elasticsearch = ElasticsearchSearchAdapter(config.elasticsearch)
        sqlite = SQLiteReadAdapter(config.sqlite)
        lifecycle = InfrastructureLifecycle()
        lifecycle.register("milvus", milvus, required=True)
        lifecycle.register("elasticsearch", elasticsearch, required=False)
        lifecycle.register("sqlite", sqlite, required=True)
    except Exception as exc:
        print(json.dumps({"status": "CLI_ERROR", "error": type(exc).__name__}))
        return 3

    exit_code = 3
    try:
        lifecycle.start()
        report = OfflineContractValidator(
            config,
            milvus=milvus,
            elasticsearch=elasticsearch,
            sqlite=sqlite,
            encoder_smoke_vectors=smoke_vectors,
        ).validate()
        print(report.model_dump_json(indent=2))
        if report.status is ValidationStatus.FAIL:
            exit_code = 2
        elif report.status is ValidationStatus.PARTIAL and args.fail_on_partial:
            exit_code = 1
        else:
            exit_code = 0
    except Exception as exc:
        print(json.dumps({"status": "CLI_ERROR", "error": type(exc).__name__}))
        exit_code = 3
    finally:
        try:
            lifecycle.close()
        except Exception as exc:
            print(json.dumps({"status": "CLI_ERROR", "error": type(exc).__name__}))
            exit_code = 3
    return exit_code


def _load_smoke_vectors(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("encoder smoke JSON could not be read") from exc
    if not isinstance(payload, dict):
        raise ValueError("encoder smoke JSON must be an object")
    factories: dict[str, Any] = {}
    for resource, vector in payload.items():
        if (
            not isinstance(resource, str)
            or not resource.strip()
            or not isinstance(vector, list)
            or not vector
        ):
            raise ValueError("encoder smoke JSON has invalid collection/vector")
        values = tuple(float(value) for value in vector)
        factories[resource] = lambda values=values: values
    return factories


if __name__ == "__main__":
    raise SystemExit(main())
