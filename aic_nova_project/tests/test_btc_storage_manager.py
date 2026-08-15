from pathlib import Path

import pytest

from scripts.btc_storage_manager import (
    build_inventory,
    changed_inventory_files,
    inspect_migration_state,
    migrate_raw_layout,
    promote_candidate,
    rollback_promotion,
    read_sqlite_video_ids,
    validate_staged_dataset,
)


def _write_video(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_inventory_records_relative_path_size_and_sha256(tmp_path):
    _write_video(tmp_path / "Videos_L21_a" / "L21_V001.mp4", b"video")

    inventory = build_inventory(tmp_path)

    assert inventory["video_count"] == 1
    assert inventory["total_bytes"] == 5
    assert inventory["files"][0]["relative_path"] == "Videos_L21_a/L21_V001.mp4"
    assert len(inventory["files"][0]["sha256"]) == 64
    assert len(inventory["aggregate_sha256"]) == 64


def test_inventory_diff_returns_only_missing_or_changed_files(tmp_path):
    video = tmp_path / "Videos_L21_a" / "L21_V001.mp4"
    _write_video(video, b"video")
    local = build_inventory(tmp_path)
    remote = {**local, "files": [dict(local["files"][0])]}
    assert changed_inventory_files(local, remote) == []

    remote["files"][0]["sha256"] = "0" * 64
    assert changed_inventory_files(local, remote) == [local["files"][0]]


def test_migration_is_resumable_when_some_batches_already_moved(tmp_path):
    _write_video(tmp_path / "Videos_L21_a" / "L21_V001.mp4", b"one")
    _write_video(
        tmp_path / "raw_videos" / "Videos_L22_a" / "L22_V001.mp4",
        b"two",
    )

    result = migrate_raw_layout(tmp_path, expected_batch_count=2)

    assert result["status"] == "complete"
    assert not (tmp_path / "Videos_L21_a").exists()
    assert (tmp_path / "raw_videos" / "Videos_L21_a").is_dir()
    assert (tmp_path / "raw_videos" / "Videos_L22_a").is_dir()

    before = __import__("json").loads(
        (tmp_path / ".migration" / "raw-inventory-before.json").read_text(
            encoding="utf-8"
        )
    )
    after = __import__("json").loads(
        (tmp_path / ".migration" / "raw-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert before["aggregate_sha256"] == after["aggregate_sha256"]
    assert before["video_count"] == after["video_count"] == 2
    assert result["pre_migration_inventory"]["aggregate_sha256"] == before[
        "aggregate_sha256"
    ]
    assert result["post_migration_inventory"]["aggregate_sha256"] == after[
        "aggregate_sha256"
    ]


def test_migration_fails_when_batch_exists_in_both_locations(tmp_path):
    _write_video(tmp_path / "Videos_L21_a" / "L21_V001.mp4", b"one")
    _write_video(
        tmp_path / "raw_videos" / "Videos_L21_a" / "L21_V001.mp4",
        b"two",
    )

    state = inspect_migration_state(tmp_path)
    assert state["Videos_L21_a"] == "ambiguous"
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_raw_layout(tmp_path, expected_batch_count=1)


def test_staged_dataset_requires_ready_manifest_and_matching_metadata(tmp_path):
    processed = tmp_path / "processed"
    metadata = processed / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "L21_V001.json").write_text("{}", encoding="utf-8")
    manifest = {
        "status": "READY",
        "dataset_id": "btc-slice",
        "dataset_fingerprint": "sha256:" + "a" * 64,
        "record_counts": {"videos": 1},
    }
    (processed / "dataset-manifest.json").write_text(
        __import__("json").dumps(manifest),
        encoding="utf-8",
    )

    validated = validate_staged_dataset(tmp_path)
    assert validated["dataset_id"] == "btc-slice"

    manifest["status"] = "BUILDING"
    (processed / "dataset-manifest.json").write_text(
        __import__("json").dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="READY"):
        validate_staged_dataset(tmp_path)


def test_publish_journal_can_roll_back_canonical_dataset(tmp_path):
    storage_root = tmp_path / "storage"
    candidate = tmp_path / "candidate"
    for root, marker in ((storage_root, "old"), (candidate, "new")):
        (root / "processed").mkdir(parents=True)
        (root / "processed" / "marker.txt").write_text(marker, encoding="utf-8")
        (root / "databases").mkdir()
        (root / "databases" / "marker.txt").write_text(marker, encoding="utf-8")
        (root / "metadata.db").write_text(marker, encoding="utf-8")

    journal = promote_candidate(storage_root, candidate, dataset_id="btc-slice")
    assert (storage_root / "processed" / "marker.txt").read_text() == "new"

    rollback_promotion(Path(journal["journal_path"]))
    assert (storage_root / "processed" / "marker.txt").read_text() == "old"


def test_legacy_video_ids_are_read_from_sqlite_without_writes(tmp_path):
    import sqlite3

    database = tmp_path / "metadata.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE videos (video_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO videos(video_id) VALUES (?)",
            [("test-b",), ("test-a",)],
        )

    assert read_sqlite_video_ids(database) == ["test-a", "test-b"]
