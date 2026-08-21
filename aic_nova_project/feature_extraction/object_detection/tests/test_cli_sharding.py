from pathlib import Path

import pytest

from src.object_detection.cli import select_metadata_files


def test_select_metadata_files_partitions_sorted_inputs(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    for video_id in ["V006", "V001", "V004", "V002", "V005", "V003"]:
        (metadata_dir / f"{video_id}.json").write_text("{}", encoding="utf-8")

    shards = [
        select_metadata_files(metadata_dir, shard_index=index, shard_count=3)
        for index in range(3)
    ]

    assert [[path.stem for path in shard] for shard in shards] == [
        ["V001", "V004"],
        ["V002", "V005"],
        ["V003", "V006"],
    ]
    assert sorted(path for shard in shards for path in shard) == sorted(
        metadata_dir.glob("*.json")
    )


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    [(-1, 2), (2, 2), (0, 0)],
)
def test_select_metadata_files_rejects_invalid_shard_configuration(
    tmp_path: Path,
    shard_index: int,
    shard_count: int,
) -> None:
    with pytest.raises(ValueError, match="shard"):
        select_metadata_files(tmp_path, shard_index, shard_count)
