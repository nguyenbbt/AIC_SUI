"""Verify that a Module 1 keyframe backup is safe to upload and consume.

The verifier is intentionally read-only. It validates metadata references and
the RIFF container length of every WebP file, which detects interrupted copies
without paying the cost of running OCR or decoding every image.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence


@dataclass(frozen=True)
class VerificationIssue:
    """One actionable backup verification finding."""

    code: str
    path: str
    message: str


@dataclass
class BackupVerificationResult:
    """Aggregate result for one backup root."""

    backup_root: str
    metadata_files: int = 0
    referenced_keyframes: int = 0
    scanned_keyframes: int = 0
    errors: list[VerificationIssue] = field(default_factory=list)
    warnings: list[VerificationIssue] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Return whether the backup passed every mandatory check."""

        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""

        return {
            "status": "PASS" if self.is_valid else "FAIL",
            "backup_root": self.backup_root,
            "metadata_files": self.metadata_files,
            "referenced_keyframes": self.referenced_keyframes,
            "scanned_keyframes": self.scanned_keyframes,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "duration_sec": round(self.duration_sec, 3),
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }


def _issue(code: str, path: Path | str, message: str) -> VerificationIssue:
    return VerificationIssue(code=code, path=str(path), message=message)


def _safe_keyframe_path(root: Path, raw_path: object) -> tuple[Path | None, str]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "Metadata file_path must be a non-empty string."

    normalized = raw_path.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"Metadata path escapes the backup root: {raw_path!r}."

    # ``root`` is already absolute and ``relative`` cannot contain ``..`` or
    # an absolute anchor. Avoid resolving every one of hundreds of thousands
    # of paths: on Windows that causes an extra filesystem lookup per frame.
    return root.joinpath(*relative.parts), ""


def _read_metadata_references(
    root: Path,
    metadata_dir: Path,
    result: BackupVerificationResult,
) -> set[Path]:
    references: set[Path] = set()
    metadata_paths = sorted(metadata_dir.glob("*.json"))
    result.metadata_files = len(metadata_paths)

    if not metadata_paths:
        result.errors.append(
            _issue("NO_METADATA", metadata_dir, "No metadata JSON files were found.")
        )
        return references

    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            result.errors.append(
                _issue("INVALID_METADATA_JSON", metadata_path, str(exc))
            )
            continue

        video_id = payload.get("video_id") if isinstance(payload, dict) else None
        if video_id != metadata_path.stem:
            result.errors.append(
                _issue(
                    "VIDEO_ID_MISMATCH",
                    metadata_path,
                    f"video_id {video_id!r} does not match file stem {metadata_path.stem!r}.",
                )
            )

        shots = payload.get("shots") if isinstance(payload, dict) else None
        if not isinstance(shots, list):
            result.errors.append(
                _issue("INVALID_SHOTS", metadata_path, "Metadata shots must be a list.")
            )
            continue

        for shot_index, shot in enumerate(shots):
            keyframes = shot.get("keyframes") if isinstance(shot, dict) else None
            if not isinstance(keyframes, list):
                result.errors.append(
                    _issue(
                        "INVALID_KEYFRAMES",
                        metadata_path,
                        f"shots[{shot_index}].keyframes must be a list.",
                    )
                )
                continue

            for keyframe_index, keyframe in enumerate(keyframes):
                raw_path = keyframe.get("file_path") if isinstance(keyframe, dict) else None
                keyframe_path, reason = _safe_keyframe_path(root, raw_path)
                if keyframe_path is None:
                    result.errors.append(
                        _issue(
                            "UNSAFE_KEYFRAME_PATH",
                            metadata_path,
                            f"shots[{shot_index}].keyframes[{keyframe_index}]: {reason}",
                        )
                    )
                    continue

                if keyframe_path in references:
                    result.errors.append(
                        _issue(
                            "DUPLICATE_KEYFRAME_REFERENCE",
                            metadata_path,
                            f"Keyframe is referenced more than once: {raw_path}",
                        )
                    )
                    continue

                references.add(keyframe_path)

    result.referenced_keyframes = len(references)
    return references


def _verify_webp(path: Path) -> VerificationIssue | None:
    try:
        actual_size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(12)
    except OSError as exc:
        return _issue("UNREADABLE_KEYFRAME", path, str(exc))

    if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WEBP":
        return _issue(
            "INVALID_WEBP_HEADER",
            path,
            "Expected a 12-byte RIFF/WEBP header.",
        )

    declared_size = int.from_bytes(header[4:8], "little") + 8
    if actual_size != declared_size:
        return _issue(
            "WEBP_SIZE_MISMATCH",
            path,
            f"Actual size is {actual_size} bytes; RIFF declares {declared_size} bytes.",
        )
    return None


def _scan_webp_tree(root: Path) -> tuple[set[Path], list[VerificationIssue]]:
    """Scan one video directory sequentially inside a bounded worker."""

    paths: set[Path] = set()
    issues: list[VerificationIssue] = []
    for path in root.rglob("*.webp"):
        paths.add(path)
        issue = _verify_webp(path)
        if issue is not None:
            issues.append(issue)
    return paths, issues


def verify_backup(
    backup_root: str | Path,
    *,
    workers: int = 8,
    progress_every: int = 0,
    progress_callback: Callable[[int], None] | None = None,
) -> BackupVerificationResult:
    """Verify metadata references and every WebP under ``backup_root``."""

    started_at = time.monotonic()
    root = Path(backup_root).expanduser().resolve(strict=False)
    result = BackupVerificationResult(backup_root=str(root))
    metadata_dir = root / "metadata"
    keyframe_dir = root / "keyframes"

    if not root.is_dir():
        result.errors.append(
            _issue("MISSING_BACKUP_ROOT", root, "Backup root does not exist.")
        )
        result.duration_sec = time.monotonic() - started_at
        return result
    if not metadata_dir.is_dir():
        result.errors.append(
            _issue("MISSING_METADATA_DIR", metadata_dir, "Metadata directory is missing.")
        )
    if not keyframe_dir.is_dir():
        result.errors.append(
            _issue("MISSING_KEYFRAME_DIR", keyframe_dir, "Keyframe directory is missing.")
        )

    references = (
        _read_metadata_references(root, metadata_dir, result)
        if metadata_dir.is_dir()
        else set()
    )

    if workers < 1:
        raise ValueError("workers must be at least 1")

    scanned_paths: set[Path] = set()
    if keyframe_dir.is_dir():
        for path in keyframe_dir.glob("*.webp"):
            scanned_paths.add(path)
            result.scanned_keyframes += 1
            webp_issue = _verify_webp(path)
            if webp_issue is not None:
                result.errors.append(webp_issue)

        video_dirs = sorted(path for path in keyframe_dir.iterdir() if path.is_dir())
        next_progress = progress_every
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for paths, issues in executor.map(_scan_webp_tree, video_dirs):
                scanned_paths.update(paths)
                result.scanned_keyframes += len(paths)
                result.errors.extend(issues)
                if progress_every > 0 and progress_callback is not None:
                    while result.scanned_keyframes >= next_progress:
                        progress_callback(result.scanned_keyframes)
                        next_progress += progress_every

    for missing in sorted(references - scanned_paths):
        result.errors.append(
            _issue(
                "MISSING_KEYFRAME",
                missing,
                "Referenced by metadata but not found in the keyframe backup.",
            )
        )

    for orphan in sorted(scanned_paths - references):
        result.warnings.append(
            _issue(
                "ORPHAN_KEYFRAME",
                orphan,
                "WebP exists but is not referenced by metadata.",
            )
        )

    result.duration_sec = time.monotonic() - started_at
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify metadata and WebP integrity in a keyframe backup."
    )
    parser.add_argument(
        "--backup-root",
        required=True,
        help="Directory containing metadata/ and keyframes/.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional destination for the complete JSON report.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum errors and warnings printed to the terminal.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of video directories scanned concurrently.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress after this many WebP files; use 0 to disable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the backup verifier CLI and return a process exit code."""

    args = _build_parser().parse_args(argv)
    if args.max_errors < 0 or args.progress_every < 0 or args.workers < 1:
        raise SystemExit(
            "--max-errors/--progress-every must be non-negative and --workers positive."
        )

    result = verify_backup(
        args.backup_root,
        workers=args.workers,
        progress_every=args.progress_every,
        progress_callback=lambda count: print(f"Scanned {count:,} WebP files..."),
    )

    report = result.to_dict()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Status: {report['status']}")
    print(f"Metadata files: {result.metadata_files:,}")
    print(f"Referenced keyframes: {result.referenced_keyframes:,}")
    print(f"Scanned keyframes: {result.scanned_keyframes:,}")
    print(f"Errors: {len(result.errors):,}")
    print(f"Warnings: {len(result.warnings):,}")
    print(f"Duration: {result.duration_sec:.2f}s")

    visible_issues = [*result.errors, *result.warnings][: args.max_errors]
    for issue in visible_issues:
        print(f"[{issue.code}] {issue.path}: {issue.message}")
    hidden_count = len(result.errors) + len(result.warnings) - len(visible_issues)
    if hidden_count > 0:
        print(f"... {hidden_count:,} additional findings; use --report for details.")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
