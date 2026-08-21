"""Upload keyframes to a Modal Volume one video at a time.

Each successful video directory is immediately visible and is verified against
the local WebP filenames. Re-running the command skips directories that already
match, which makes interrupted uploads safely resumable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Any, Sequence


DEFAULT_REMOTE_PARENT = "/processed/keyframes"


def discover_keyframe_directories(root: Path) -> tuple[list[Path], int]:
    """Return sorted non-empty video directories and their total WebP count."""
    if not root.is_dir():
        raise FileNotFoundError(f"Keyframe root does not exist: {root}")

    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not directories:
        raise ValueError(f"Keyframe root contains no video directories: {root}")

    total_files = 0
    for directory in directories:
        file_count = sum(1 for _ in directory.glob("*.webp"))
        if file_count == 0:
            raise ValueError(
                f"Video directory contains no WebP files: {directory}"
            )
        total_files += file_count
    return directories, total_files


def parse_remote_webp_names(entries: Sequence[dict[str, Any]]) -> set[str]:
    """Extract direct WebP basenames from a Modal ``volume ls --json`` result."""
    return {
        PurePosixPath(str(entry["filename"])).name
        for entry in entries
        if entry.get("type") == "file"
        and str(entry.get("filename", "")).lower().endswith(".webp")
    }


def build_put_command(
    modal_executable: Path,
    volume_name: str,
    local_directory: Path,
    remote_parent: str,
) -> list[str]:
    """Build a shell-free command that preserves the local video directory."""
    normalized_parent = f"/{remote_parent.strip('/')}" if remote_parent.strip("/") else ""
    if not normalized_parent:
        raise ValueError("Remote parent must not be the Volume root")
    return [
        str(modal_executable),
        "volume",
        "put",
        "--force",
        volume_name,
        str(local_directory),
        f"{normalized_parent}/",
    ]


def _local_webp_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.webp")}


def _remote_webp_names(
    modal_executable: Path,
    volume_name: str,
    remote_path: str,
    environment: dict[str, str],
) -> set[str] | None:
    completed = subprocess.run(
        [
            str(modal_executable),
            "volume",
            "ls",
            volume_name,
            remote_path,
            "--json",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        entries = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Modal returned invalid JSON for {remote_path}: {exc}"
        ) from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected Modal listing for {remote_path}")
    return parse_remote_webp_names(entries)


def upload_keyframes(
    *,
    modal_executable: Path,
    profile: str,
    volume_name: str,
    local_root: Path,
    remote_parent: str,
    expected_video_count: int,
    expected_file_count: int,
    attempts: int,
) -> None:
    """Upload, verify, and resume a keyframe dataset one video at a time."""
    directories, total_files = discover_keyframe_directories(local_root)
    if len(directories) != expected_video_count:
        raise ValueError(
            f"Local video count is {len(directories)}, expected "
            f"{expected_video_count}"
        )
    if total_files != expected_file_count:
        raise ValueError(
            f"Local WebP count is {total_files}, expected {expected_file_count}"
        )
    if attempts < 1:
        raise ValueError("Attempts must be at least 1")

    environment = os.environ.copy()
    environment["MODAL_PROFILE"] = profile
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"

    verified_files = 0
    started_at = time.monotonic()
    for index, directory in enumerate(directories, start=1):
        local_names = _local_webp_names(directory)
        remote_path = f"/{remote_parent.strip('/')}/{directory.name}"
        remote_names = _remote_webp_names(
            modal_executable,
            volume_name,
            remote_path,
            environment,
        )
        if remote_names == local_names:
            status = "SKIP"
        else:
            status = "UPLOAD"
            command = build_put_command(
                modal_executable,
                volume_name,
                directory,
                remote_parent,
            )
            for attempt in range(1, attempts + 1):
                print(
                    f"[{index:03d}/{len(directories)}] {directory.name}: "
                    f"uploading {len(local_names):,} files "
                    f"(attempt {attempt}/{attempts})",
                    flush=True,
                )
                completed = subprocess.run(
                    command,
                    check=False,
                    env=environment,
                )
                if completed.returncode == 0:
                    remote_names = _remote_webp_names(
                        modal_executable,
                        volume_name,
                        remote_path,
                        environment,
                    )
                    if remote_names == local_names:
                        break
                if attempt < attempts:
                    time.sleep(min(5 * attempt, 15))
            else:
                remote_count = 0 if remote_names is None else len(remote_names)
                raise RuntimeError(
                    f"Verification failed for {directory.name}: "
                    f"local={len(local_names)}, remote={remote_count}"
                )

        verified_files += len(local_names)
        elapsed_minutes = (time.monotonic() - started_at) / 60
        percent = 100 * index / len(directories)
        print(
            f"[{index:03d}/{len(directories)}] {directory.name}: {status} PASS; "
            f"verified={verified_files:,}/{total_files:,} "
            f"({percent:.1f}%, {elapsed_minutes:.1f} min)",
            flush=True,
        )

    print(
        f"UPLOAD COMPLETE: {len(directories):,} videos, "
        f"{verified_files:,} WebP files verified.",
        flush=True,
    )


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--remote-parent", default=DEFAULT_REMOTE_PARENT)
    parser.add_argument("--expected-video-count", type=int, required=True)
    parser.add_argument("--expected-file-count", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--modal-executable",
        type=Path,
        default=Path("modal"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    try:
        upload_keyframes(
            modal_executable=args.modal_executable,
            profile=args.profile,
            volume_name=args.volume,
            local_root=args.local_root,
            remote_parent=args.remote_parent,
            expected_video_count=args.expected_video_count,
            expected_file_count=args.expected_file_count,
            attempts=args.attempts,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
