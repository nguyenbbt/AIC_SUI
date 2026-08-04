from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compose_is_compatible_with_mgpux_sandbox_networking() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    services = compose["services"]

    assert "version" not in compose
    assert 2379 in services["etcd"]["expose"]
    assert 9000 in services["minio"]["expose"]

    elasticsearch_build = services["elasticsearch"]["build"]
    elasticsearch_dockerfile = (
        PROJECT_ROOT / elasticsearch_build["dockerfile"]
    )
    assert elasticsearch_dockerfile.is_file()

    indexing = services["indexing"]
    assert "./data:/workspace/data" in indexing["volumes"]
    assert indexing["environment"]["MILVUS_URI"].endswith(
        "milvus-standalone:19530"
    )
    assert indexing["environment"]["ES_URI"].endswith(
        "elasticsearch:9200"
    )


def test_vscode_recommends_the_mgpux_extension() -> None:
    extensions = yaml.safe_load(
        (PROJECT_ROOT / ".vscode" / "extensions.json").read_text(
            encoding="utf-8"
        )
    )
    assert "puxpux.m-gpux" in extensions["recommendations"]
