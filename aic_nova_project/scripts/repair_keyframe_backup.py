"""Restore missing or truncated keyframes from raw videos and Module 1 metadata.

The verification report is the repair allow-list. Dry-run is the default;
``--apply`` is required to atomically replace damaged files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from uuid import uuid4

from scripts.verify_keyframe_backup import _verify_webp


REPAIRABLE_CODES = {
    "MISSING_KEYFRAME",
    "INVALID_WEBP_HEADER",
    "WEBP_SIZE_MISMATCH",
    "UNREADABLE_KEYFRAME",
}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm"}


@dataclass(frozen=True, order=True)
class RepairTarget:
    """One keyframe that must be reconstructed from a raw frame."""

    output_path: Path
    frame_index: int


@dataclass(frozen=True)
class VideoRepairResult:
    """Result of repairing one video directory."""

    video_id: str
    repaired: int
    skipped: int
    duration_sec: float


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def _metadata_frame_map(backup_root: Path, video_id: str) -> dict[Path, int]:
    metadata_path = backup_root / "metadata" / f"{video_id}.json"
    payload = _load_json(metadata_path)
    if not isinstance(payload, dict) or payload.get("video_id") != video_id:
        raise ValueError(f"Invalid video_id contract in {metadata_path}")

    frame_map: dict[Path, int] = {}
    shots = payload.get("shots")
    if not isinstance(shots, list):
        raise ValueError(f"Metadata shots must be a list: {metadata_path}")

    for shot in shots:
        keyframes = shot.get("keyframes") if isinstance(shot, dict) else None
        if not isinstance(keyframes, list):
            raise ValueError(f"Metadata keyframes must be a list: {metadata_path}")
        for keyframe in keyframes:
            if not isinstance(keyframe, dict):
                raise ValueError(f"Invalid keyframe record: {metadata_path}")
            raw_path = keyframe.get("file_path")
            frame_index = keyframe.get("frame_index")
            if not isinstance(raw_path, str) or not isinstance(frame_index, int):
                raise ValueError(f"Missing file_path/frame_index: {metadata_path}")
            relative = PurePosixPath(raw_path.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe metadata keyframe path: {raw_path!r}")
            output_path = backup_root.joinpath(*relative.parts)
            if output_path in frame_map:
                raise ValueError(f"Duplicate metadata keyframe path: {raw_path}")
            frame_map[output_path] = frame_index
    return frame_map


def build_repair_plan(
    report_path: str | Path,
    backup_root: str | Path,
) -> dict[str, list[RepairTarget]]:
    """Build a fail-closed repair plan from a verifier JSON report."""

    root = Path(backup_root).expanduser().resolve(strict=False)
    keyframe_root = root / "keyframes"
    report = _load_json(Path(report_path))
    errors = report.get("errors") if isinstance(report, dict) else None
    if not isinstance(errors, list):
        raise ValueError("Verification report must contain an errors list")

    paths_by_video: dict[str, set[Path]] = {}
    for error in errors:
        if not isinstance(error, dict) or error.get("code") not in REPAIRABLE_CODES:
            continue
        raw_path = error.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("Repairable report issue has no path")
        output_path = Path(raw_path).expanduser().resolve(strict=False)
        try:
            relative = output_path.relative_to(keyframe_root)
        except ValueError as exc:
            raise ValueError(
                f"Report path is outside the keyframe backup: {output_path}"
            ) from exc
        if len(relative.parts) < 2:
            raise ValueError(f"Report path has no video directory: {output_path}")
        paths_by_video.setdefault(relative.parts[0], set()).add(output_path)

    plan: dict[str, list[RepairTarget]] = {}
    for video_id in sorted(paths_by_video):
        frame_map = _metadata_frame_map(root, video_id)
        targets: list[RepairTarget] = []
        for output_path in sorted(paths_by_video[video_id]):
            if output_path not in frame_map:
                raise ValueError(
                    f"Report keyframe is not present in metadata: {output_path}"
                )
            targets.append(
                RepairTarget(
                    output_path=output_path,
                    frame_index=frame_map[output_path],
                )
            )
        plan[video_id] = sorted(targets, key=lambda item: (item.frame_index, item.output_path))

    if not plan:
        raise ValueError("Verification report contains no repairable keyframe errors")
    return plan


def build_raw_video_index(raw_root: str | Path) -> dict[str, Path]:
    """Index raw videos recursively and reject duplicate basenames."""

    root = Path(raw_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"Raw video root does not exist: {root}")

    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        video_id = path.stem
        if video_id in index:
            raise ValueError(
                f"Duplicate raw video_id {video_id!r}: {index[video_id]} and {path}"
            )
        index[video_id] = path
    return index


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_video(backup_root: Path, video_id: str, video_path: Path) -> None:
    """Prove that frame indices were produced from this exact raw video."""

    metadata_path = backup_root / "metadata" / f"{video_id}.json"
    payload = _load_json(metadata_path)
    expected = payload.get("source_fingerprint") if isinstance(payload, dict) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing source_fingerprint in {metadata_path}")
    actual = _sha256(video_path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"Raw video fingerprint mismatch for {video_id}: "
            f"expected {expected}, got {actual}"
        )


def _read_frame(
    capture: Any,
    frame_index: int,
    next_frame_index: int | None,
    cv2_module: Any,
    seek_threshold: int,
) -> tuple[Any, int]:
    if (
        next_frame_index is None
        or frame_index < next_frame_index
        or frame_index - next_frame_index > seek_threshold
    ):
        if not capture.set(cv2_module.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(f"Cannot seek to frame {frame_index}")
        next_frame_index = frame_index

    while next_frame_index < frame_index:
        if not capture.grab():
            raise RuntimeError(f"Cannot advance to frame {frame_index}")
        next_frame_index += 1

    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot decode frame {frame_index}")
    return frame, frame_index + 1


def _atomic_write_webp(
    output_path: Path,
    frame: Any,
    webp_quality: int,
    cv2_module: Any,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.repair-{uuid4().hex}.webp"
    )
    try:
        saved = cv2_module.imwrite(
            str(temporary_path),
            frame,
            [cv2_module.IMWRITE_WEBP_QUALITY, webp_quality],
        )
        if not saved:
            raise RuntimeError(f"OpenCV could not save {temporary_path}")
        issue = _verify_webp(temporary_path)
        if issue is not None:
            raise RuntimeError(f"Repaired WebP failed verification: {issue.message}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def repair_video(
    video_path: Path,
    targets: Sequence[RepairTarget],
    webp_quality: int,
    cv2_module: Any,
    *,
    seek_threshold: int = 250,
) -> VideoRepairResult:
    """Atomically restore every damaged keyframe for one video."""

    started_at = time.monotonic()
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open raw video: {video_path}")

    repaired = 0
    skipped = 0
    next_frame_index: int | None = None
    cached_index: int | None = None
    cached_frame: Any = None
    try:
        for target in sorted(targets, key=lambda item: (item.frame_index, item.output_path)):
            if target.output_path.is_file() and _verify_webp(target.output_path) is None:
                skipped += 1
                continue
            if target.frame_index == cached_index:
                frame = cached_frame
            else:
                frame, next_frame_index = _read_frame(
                    capture,
                    target.frame_index,
                    next_frame_index,
                    cv2_module,
                    seek_threshold,
                )
                cached_index = target.frame_index
                cached_frame = frame
            _atomic_write_webp(
                target.output_path,
                frame,
                webp_quality,
                cv2_module,
            )
            repaired += 1
    finally:
        capture.release()

    return VideoRepairResult(
        video_id=video_path.stem,
        repaired=repaired,
        skipped=skipped,
        duration_sec=time.monotonic() - started_at,
    )


def _write_journal(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore verifier-approved keyframes from raw videos."
    )
    parser.add_argument("--backup-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run dry-run planning or apply the exact repair allow-list."""

    args = _build_parser().parse_args(argv)
    if args.workers < 1 or not 0 <= args.quality <= 100:
        raise SystemExit("--workers must be positive and --quality must be 0..100")

    backup_root = Path(args.backup_root).expanduser().resolve(strict=False)
    plan = build_repair_plan(args.report, backup_root)
    raw_index = build_raw_video_index(args.raw_root)
    missing_raw = sorted(set(plan) - set(raw_index))
    if missing_raw:
        raise SystemExit(f"Missing raw videos: {', '.join(missing_raw)}")

    target_count = sum(len(targets) for targets in plan.values())
    print(f"Videos to repair: {len(plan):,}")
    print(f"Keyframes to repair: {target_count:,}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply after reviewing this plan.")
        return 0

    import cv2  # Imported only for an approved mutation run.

    journal: dict[str, Any] = {
        "status": "RUNNING",
        "backup_root": str(backup_root),
        "raw_root": str(Path(args.raw_root).expanduser().resolve(strict=False)),
        "report": str(Path(args.report).expanduser().resolve(strict=False)),
        "video_count": len(plan),
        "target_count": target_count,
        "videos": {},
    }
    _write_journal(args.journal, journal)

    def run_video(video_id: str) -> VideoRepairResult:
        video_path = raw_index[video_id]
        validate_source_video(backup_root, video_id, video_path)
        return repair_video(
            video_path=video_path,
            targets=plan[video_id],
            webp_quality=args.quality,
            cv2_module=cv2,
        )

    failures = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_video, video_id): video_id for video_id in plan
        }
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                result = future.result()
                journal["videos"][video_id] = {
                    "status": "PASS",
                    **asdict(result),
                }
                print(
                    f"[{completed + 1}/{len(plan)}] {video_id}: "
                    f"repaired={result.repaired}, skipped={result.skipped}"
                )
            except Exception as exc:  # noqa: BLE001 - journal every video failure
                failures += 1
                journal["videos"][video_id] = {
                    "status": "FAIL",
                    "error": str(exc),
                }
                print(f"[{completed + 1}/{len(plan)}] {video_id}: FAIL: {exc}")
            completed += 1
            _write_journal(args.journal, journal)

    journal["status"] = "PASS" if failures == 0 else "FAIL"
    journal["failure_count"] = failures
    _write_journal(args.journal, journal)
    print(f"Repair status: {journal['status']}; failed videos: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
