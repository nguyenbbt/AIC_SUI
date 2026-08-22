from src.indexing.cli import build_parser


def test_indexing_cli_accepts_backend_contract_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DATA_DIR", "/workspace/data/processed")
    monkeypatch.setenv("MILVUS_URI", "http://milvus-tunnel:19530")
    monkeypatch.setenv("ES_URI", "http://elasticsearch-tunnel:9200")
    monkeypatch.setenv(
        "DB_URI",
        "sqlite:////workspace/data/metadata.db",
    )

    args = build_parser().parse_args([])

    assert args.data_dir.as_posix() == "/workspace/data/processed"
    assert args.milvus_uri == "http://milvus-tunnel:19530"
    assert args.es_uri == "http://elasticsearch-tunnel:9200"
    assert args.db_uri == "sqlite:////workspace/data/metadata.db"


def test_indexing_cli_accepts_repeatable_video_repairs(monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", "/workspace/data/processed")

    args = build_parser().parse_args(
        [
            "--video-id",
            "L26_V308",
            "--video-id",
            "L26_V309",
            "--finalize",
            "--unpublished-repair",
        ]
    )

    assert args.video_ids == ["L26_V308", "L26_V309"]
    assert args.finalize is True
    assert args.unpublished_repair is True
