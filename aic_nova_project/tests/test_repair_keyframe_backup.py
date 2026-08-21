from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.repair_keyframe_backup import (
    RepairTarget,
    build_repair_plan,
    repair_video,
)
from scripts.verify_keyframe_backup import _verify_webp


def _write_metadata(root: Path, video_id: str, frame_index: int) -> Path:
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{video_id}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "shots": [
                    {
                        "shot_id": 0,
                        "keyframes": [
                            {
                                "file_path": (
                                    f"keyframes/{video_id}/shot_00000_pos_015.webp"
                                ),
                                "frame_index": frame_index,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


def test_build_repair_plan_maps_report_path_to_metadata_frame(tmp_path: Path) -> None:
    backup_root = tmp_path / "backup"
    _write_metadata(backup_root, "V001", frame_index=37)
    damaged_path = (
        backup_root / "keyframes" / "V001" / "shot_00000_pos_015.webp"
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "errors": [
                    {
                        "code": "MISSING_KEYFRAME",
                        "path": str(damaged_path),
                        "message": "missing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_repair_plan(report_path, backup_root)

    assert list(plan) == ["V001"]
    assert plan["V001"] == [
        RepairTarget(output_path=damaged_path, frame_index=37)
    ]


def test_build_repair_plan_rejects_report_path_outside_backup(tmp_path: Path) -> None:
    backup_root = tmp_path / "backup"
    _write_metadata(backup_root, "V001", frame_index=37)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "errors": [
                    {
                        "code": "MISSING_KEYFRAME",
                        "path": str(tmp_path / "outside.webp"),
                        "message": "missing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        build_repair_plan(report_path, backup_root)
    except ValueError as exc:
        assert "outside the keyframe backup" in str(exc)
    else:
        raise AssertionError("Unsafe report path was accepted")


class _FakeCapture:
    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self.frames = frames
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV-compatible API
        return True

    def set(self, _property: int, value: int) -> bool:
        self.position = int(value)
        return True

    def grab(self) -> bool:
        if self.position not in self.frames:
            return False
        self.position += 1
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self.frames.get(self.position)
        self.position += 1
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


class _FakeCv2:
    CAP_PROP_POS_FRAMES = 1
    IMWRITE_WEBP_QUALITY = 64

    def __init__(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.capture = _FakeCapture({37: frame, 38: frame})

    def VideoCapture(self, _path: str) -> _FakeCapture:  # noqa: N802
        return self.capture

    @staticmethod
    def imwrite(path: str, _frame: np.ndarray, _params: list[int]) -> bool:
        payload = b"WEBP" + b"repaired"
        Path(path).write_bytes(b"RIFF" + len(payload).to_bytes(4, "little") + payload)
        return True


def test_repair_video_atomically_restores_damaged_webp(tmp_path: Path) -> None:
    output = tmp_path / "keyframes" / "V001" / "shot_00000_pos_015.webp"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"truncated")
    fake_cv2 = _FakeCv2()

    result = repair_video(
        video_path=tmp_path / "V001.mp4",
        targets=[RepairTarget(output_path=output, frame_index=37)],
        webp_quality=90,
        cv2_module=fake_cv2,
    )

    assert result.repaired == 1
    assert result.skipped == 0
    assert _verify_webp(output) is None
    assert fake_cv2.capture.released
    assert not list(output.parent.glob(".*.repair-*.webp"))
