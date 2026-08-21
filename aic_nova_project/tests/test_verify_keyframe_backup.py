from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_keyframe_backup import main, verify_backup


def _write_webp(path: Path, payload: bytes = b"payload") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"WEBP" + payload
    path.write_bytes(b"RIFF" + len(body).to_bytes(4, "little") + body)


def _write_metadata(root: Path, video_id: str, file_paths: list[str]) -> None:
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "shots": [
            {
                "shot_id": 0,
                "keyframes": [
                    {"file_path": file_path, "frame_index": index}
                    for index, file_path in enumerate(file_paths)
                ],
            }
        ],
    }
    (metadata_dir / f"{video_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_verify_backup_accepts_complete_metadata_and_webp(tmp_path: Path) -> None:
    root = tmp_path / "backup"
    relative_path = "keyframes/V001/shot_00000_pos_015.webp"
    _write_webp(root / relative_path)
    _write_metadata(root, "V001", [relative_path])

    result = verify_backup(root)

    assert result.is_valid
    assert result.metadata_files == 1
    assert result.referenced_keyframes == 1
    assert result.scanned_keyframes == 1
    assert result.errors == []


def test_verify_backup_reports_truncated_webp(tmp_path: Path) -> None:
    root = tmp_path / "backup"
    relative_path = "keyframes/V001/shot_00000_pos_015.webp"
    image_path = root / relative_path
    _write_webp(image_path, payload=b"complete-payload")
    image_path.write_bytes(image_path.read_bytes()[:-5])
    _write_metadata(root, "V001", [relative_path])

    result = verify_backup(root)

    assert not result.is_valid
    assert any(error.code == "WEBP_SIZE_MISMATCH" for error in result.errors)


def test_verify_backup_scans_video_directories_with_bounded_workers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "backup"
    first_path = "keyframes/V001/shot_00000_pos_015.webp"
    second_path = "keyframes/V002/shot_00000_pos_015.webp"
    _write_webp(root / first_path)
    _write_webp(root / second_path)
    _write_metadata(root, "V001", [first_path])
    _write_metadata(root, "V002", [second_path])

    result = verify_backup(root, workers=2)

    assert result.is_valid
    assert result.metadata_files == 2
    assert result.scanned_keyframes == 2
    assert result.referenced_keyframes == 2


def test_verify_backup_reports_missing_and_unsafe_metadata_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "backup"
    _write_metadata(
        root,
        "V001",
        ["keyframes/V001/missing.webp", "../outside.webp"],
    )

    result = verify_backup(root)

    assert not result.is_valid
    assert {error.code for error in result.errors} >= {
        "MISSING_KEYFRAME",
        "UNSAFE_KEYFRAME_PATH",
    }


def test_cli_writes_json_report_and_returns_failure(tmp_path: Path) -> None:
    root = tmp_path / "backup"
    relative_path = "keyframes/V001/broken.webp"
    image_path = root / relative_path
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-a-webp")
    _write_metadata(root, "V001", [relative_path])
    report_path = tmp_path / "verification.json"

    exit_code = main(
        [
            "--backup-root",
            str(root),
            "--report",
            str(report_path),
            "--max-errors",
            "5",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["errors"][0]["code"] == "INVALID_WEBP_HEADER"
