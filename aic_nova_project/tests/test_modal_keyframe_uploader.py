from pathlib import Path

import pytest

from scripts.modal_keyframe_uploader import (
    build_put_command,
    discover_keyframe_directories,
    parse_remote_webp_names,
)


def test_discover_keyframe_directories_returns_sorted_inventory(
    tmp_path: Path,
) -> None:
    for video_id, file_count in (("L22_V002", 1), ("L21_V001", 2)):
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        for index in range(file_count):
            (video_dir / f"frame-{index}.webp").write_bytes(b"webp")

    directories, file_count = discover_keyframe_directories(tmp_path)

    assert [path.name for path in directories] == ["L21_V001", "L22_V002"]
    assert file_count == 3


def test_discover_keyframe_directories_rejects_empty_video_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "L21_V001").mkdir()

    with pytest.raises(ValueError, match="contains no WebP files"):
        discover_keyframe_directories(tmp_path)


def test_parse_remote_webp_names_uses_file_basenames_only() -> None:
    entries = [
        {"filename": "processed/keyframes/L21_V001/a.webp", "type": "file"},
        {"filename": "processed/keyframes/L21_V001/b.webp", "type": "file"},
        {"filename": "processed/keyframes/L21_V001/note.txt", "type": "file"},
        {"filename": "processed/keyframes/L21_V001/nested", "type": "dir"},
    ]

    assert parse_remote_webp_names(entries) == {"a.webp", "b.webp"}


def test_build_put_command_uploads_directory_under_remote_parent() -> None:
    command = build_put_command(
        modal_executable=Path("modal.exe"),
        volume_name="dataset-volume",
        local_directory=Path("E:/backup/keyframes/L21_V001"),
        remote_parent="/processed/keyframes",
    )

    assert command == [
        "modal.exe",
        "volume",
        "put",
        "--force",
        "dataset-volume",
        "E:\\backup\\keyframes\\L21_V001",
        "/processed/keyframes/",
    ]
