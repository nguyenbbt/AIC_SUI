"""Fail-closed helpers for moving and validating the BTC dataset on one SSD."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".webm"})
BATCH_PREFIX = "Videos_L"


def build_backup_online_layout(
    backup_root: Path,
    videos_root: Path,
) -> dict[str, Path]:
    """Map the backup artifact tree onto the canonical processed contract.

    The returned paths describe directory junctions.  This function never
    creates, moves, copies, or removes data.
    """
    return {
        "processed/metadata": backup_root / "metadata",
        "processed/keyframes": backup_root / "keyframes",
        "processed/videos": videos_root,
        "processed/embeddings/visual": backup_root / "embeddings" / "visual",
        "processed/embeddings/text_asr": (
            backup_root / "module6-local" / "text_asr"
        ),
        "processed/embeddings/text_ocr": (
            backup_root / "module6-local" / "text_ocr"
        ),
        "processed/embeddings/text_summary": (
            backup_root / "module6-local" / "text_summary"
        ),
        "processed/transcripts": (
            backup_root / "asr-final" / "transcripts" / "transcripts"
        ),
        "processed/summaries": (
            backup_root / "asr-final" / "summaries" / "summaries"
        ),
        "processed/ocr": backup_root / "ocr",
        "processed/object_detection": backup_root / "object_detection",
    }


def _artifact_ids(
    directory: Path,
    pattern: str,
    *,
    suffix_to_remove: str = "",
) -> set[str]:
    if not directory.is_dir():
        raise ValueError(f"Artifact directory is missing: {directory}")
    identifiers = set()
    for path in directory.glob(pattern):
        identifier = path.stem
        if suffix_to_remove:
            if not identifier.endswith(suffix_to_remove):
                continue
            identifier = identifier[: -len(suffix_to_remove)]
        if identifier:
            identifiers.add(identifier)
    return identifiers


def _assert_same_video_ids(
    family: str,
    actual: set[str],
    expected: set[str],
    *,
    allow_missing: bool = False,
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if extra or (missing and not allow_missing):
        details = []
        if missing:
            details.append(f"missing={','.join(missing[:10])}")
        if extra:
            details.append(f"extra={','.join(extra[:10])}")
        detail_text = "; ".join(details)
        raise ValueError(
            f"{family} video IDs do not match metadata: {detail_text}"
        )


def _validate_object_artifacts(
    directory: Path,
    object_ids: set[str],
) -> None:
    for video_id in sorted(object_ids):
        path = directory / f"{video_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Object Detection JSON: {path}") from exc
        if not isinstance(payload, dict) or payload.get("video_id") != video_id:
            raise ValueError(
                f"Object Detection video_id mismatch in {path}"
            )
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"Object Detection frames must be a list: {path}")
        for index, frame in enumerate(frames):
            if (
                not isinstance(frame, dict)
                or not isinstance(frame.get("frame_id"), str)
                or not isinstance(frame.get("objects"), list)
            ):
                raise ValueError(
                    f"Invalid Object Detection frame {index} in {path}"
                )


def audit_backup_online_artifacts(
    backup_root: Path,
    videos_root: Path,
    *,
    expected_video_count: int,
    expected_keyframe_count: int,
    require_objects: bool,
) -> dict[str, Any]:
    """Validate backup artifacts before exposing them to M7 or Online.

    Object Detection may remain incomplete during the preparation phase.  The
    indexing gate only opens when ``require_objects`` is true and every object
    artifact is present and structurally valid.
    """
    backup_root = backup_root.resolve(strict=True)
    videos_root = videos_root.resolve(strict=True)
    if expected_video_count < 1 or expected_keyframe_count < 1:
        raise ValueError("Expected video/keyframe counts must be positive")

    layout = build_backup_online_layout(backup_root, videos_root)
    metadata_ids = _artifact_ids(backup_root / "metadata", "*.json")
    if len(metadata_ids) != expected_video_count:
        raise ValueError(
            f"metadata count {len(metadata_ids)} != {expected_video_count}"
        )

    keyframe_root = backup_root / "keyframes"
    if not keyframe_root.is_dir():
        raise ValueError(f"Artifact directory is missing: {keyframe_root}")
    keyframe_ids = {
        path.name for path in keyframe_root.iterdir() if path.is_dir()
    }
    keyframe_count = sum(1 for _ in keyframe_root.rglob("*.webp"))
    if keyframe_count != expected_keyframe_count:
        raise ValueError(
            f"keyframes count {keyframe_count} != {expected_keyframe_count}"
        )

    video_paths = list(_iter_video_files(videos_root))
    video_ids: set[str] = set()
    for video_path in video_paths:
        if video_path.stem in video_ids:
            raise ValueError(f"Duplicate raw video_id {video_path.stem}")
        video_ids.add(video_path.stem)
    if len(video_paths) != expected_video_count:
        raise ValueError(
            f"videos count {len(video_paths)} != {expected_video_count}"
        )
    families = {
        "keyframe_directories": keyframe_ids,
        "videos": video_ids,
        "visual": _artifact_ids(
            backup_root / "embeddings" / "visual", "*.parquet"
        ),
        "transcripts": _artifact_ids(
            backup_root / "asr-final" / "transcripts" / "transcripts",
            "*_cleaned.json",
            suffix_to_remove="_cleaned",
        ),
        "summaries": _artifact_ids(
            backup_root / "asr-final" / "summaries" / "summaries",
            "*.json",
        ),
        "ocr": _artifact_ids(backup_root / "ocr", "*.json"),
        "text_asr": _artifact_ids(
            backup_root / "module6-local" / "text_asr", "*.parquet"
        ),
        "text_ocr": _artifact_ids(
            backup_root / "module6-local" / "text_ocr", "*.parquet"
        ),
        "text_summary": _artifact_ids(
            backup_root / "module6-local" / "text_summary", "*.parquet"
        ),
    }
    for family, identifiers in families.items():
        _assert_same_video_ids(family, identifiers, metadata_ids)

    object_root = backup_root / "object_detection"
    object_ids = (
        _artifact_ids(object_root, "*.json") if object_root.is_dir() else set()
    )
    _assert_same_video_ids(
        "objects",
        object_ids,
        metadata_ids,
        allow_missing=not require_objects,
    )
    _validate_object_artifacts(object_root, object_ids)
    objects_complete = object_ids == metadata_ids
    if require_objects and not objects_complete:
        raise ValueError(
            f"objects count {len(object_ids)} != {expected_video_count}"
        )

    family_counts = {
        name: len(ids)
        for name, ids in families.items()
        if name != "keyframe_directories"
    }
    artifact_counts = {
        "metadata": len(metadata_ids),
        "keyframe_video_directories": len(keyframe_ids),
        "keyframes": keyframe_count,
        **family_counts,
        "objects": len(object_ids),
    }
    return {
        "backup_root": str(backup_root),
        "videos_root": str(videos_root),
        "artifact_counts": artifact_counts,
        "video_ids": sorted(metadata_ids),
        "ready_for_indexing": require_objects and objects_complete,
        "links": {
            relative_path: str(source.resolve(strict=False))
            for relative_path, source in layout.items()
            if relative_path != "processed/object_detection" or objects_complete
        },
    }


def _iter_video_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(root: Path) -> dict[str, Any]:
    """Return a deterministic inventory and reject duplicate video IDs."""
    root = root.resolve(strict=True)
    files = [(path, path.relative_to(root).as_posix()) for path in _iter_video_files(root)]
    return _build_inventory(root, files)


def _build_inventory(
    root: Path,
    paths: list[tuple[Path, str]],
) -> dict[str, Any]:
    """Build an inventory from physical paths and stable logical paths."""
    files: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    total_bytes = 0
    for path, relative_path in sorted(paths, key=lambda item: item[1]):
        video_id = path.stem
        previous = ids.get(video_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate video_id {video_id!r}: {previous!r} and "
                f"{relative_path!r}"
            )
        ids[video_id] = relative_path
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": size,
                "sha256": _sha256(path),
            }
        )
    aggregate = hashlib.sha256()
    for entry in files:
        aggregate.update(
            (
                f"{entry['relative_path']}\0{entry['size_bytes']}\0"
                f"{entry['sha256']}\n"
            ).encode("utf-8")
        )
    return {
        "schema_version": 1,
        "root": str(root),
        "video_count": len(files),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }


def _build_batch_inventory(
    storage_root: Path,
    state: dict[str, str],
) -> dict[str, Any]:
    """Inventory mixed source/destination batches under one logical raw root."""
    paths: list[tuple[Path, str]] = []
    for batch_name, location in sorted(state.items()):
        batch_root = (
            storage_root / batch_name
            if location == "source"
            else storage_root / "raw_videos" / batch_name
        )
        for path in _iter_video_files(batch_root):
            logical_path = (Path(batch_name) / path.relative_to(batch_root)).as_posix()
            paths.append((path, logical_path))
    return _build_inventory(storage_root / "raw_videos", paths)


def changed_inventory_files(
    local: dict[str, Any],
    remote: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return local entries absent or fingerprint-mismatched remotely."""
    remote_by_path = {
        entry.get("relative_path"): entry
        for entry in (remote or {}).get("files", [])
        if isinstance(entry, dict)
    }
    changed = []
    for entry in local.get("files", []):
        previous = remote_by_path.get(entry["relative_path"])
        if previous is None or any(
            previous.get(field) != entry[field]
            for field in ("size_bytes", "sha256")
        ):
            changed.append(entry)
    return changed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact atomically."""
    _write_journal(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def read_sqlite_video_ids(database: Path) -> list[str]:
    """Read canonical video IDs through a fail-closed SQLite read-only URI."""
    database = database.resolve(strict=True)
    uri = f"file:{database.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT video_id FROM videos ORDER BY video_id"
        ).fetchall()
    return [str(row[0]) for row in rows]


def inspect_migration_state(storage_root: Path) -> dict[str, str]:
    """Classify each batch without guessing after an interrupted move."""
    storage_root = storage_root.resolve(strict=True)
    raw_root = storage_root / "raw_videos"
    source_names = {
        path.name
        for path in storage_root.iterdir()
        if path.is_dir() and path.name.startswith(BATCH_PREFIX)
    }
    destination_names = (
        {
            path.name
            for path in raw_root.iterdir()
            if path.is_dir() and path.name.startswith(BATCH_PREFIX)
        }
        if raw_root.is_dir()
        else set()
    )
    state: dict[str, str] = {}
    for name in sorted(source_names | destination_names):
        if name in source_names and name in destination_names:
            state[name] = "ambiguous"
        elif name in source_names:
            state[name] = "source"
        else:
            state[name] = "destination"
    return state


def _write_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def migrate_raw_layout(
    storage_root: Path,
    *,
    expected_batch_count: int = 14,
) -> dict[str, Any]:
    """Resume same-volume batch moves using an atomic JSON journal."""
    storage_root = storage_root.resolve(strict=True)
    state = inspect_migration_state(storage_root)
    if len(state) != expected_batch_count:
        raise ValueError(
            f"Expected {expected_batch_count} BTC batches, found {len(state)}"
        )
    ambiguous = [name for name, location in state.items() if location == "ambiguous"]
    if ambiguous:
        raise ValueError(f"Migration state is ambiguous for: {', '.join(ambiguous)}")

    raw_root = storage_root / "raw_videos"
    journal_path = storage_root / ".migration" / "raw-layout.json"
    inventory_before_path = (
        storage_root / ".migration" / "raw-inventory-before.json"
    )
    inventory_after_path = storage_root / ".migration" / "raw-inventory.json"
    if inventory_before_path.is_file():
        inventory_before = read_json(inventory_before_path)
    else:
        inventory_before = _build_batch_inventory(storage_root, state)
        write_json(inventory_before_path, inventory_before)
    raw_root.mkdir(exist_ok=True)
    journal: dict[str, Any] = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "status": "moving",
        "pre_migration_inventory": {
            "path": str(inventory_before_path),
            "video_count": inventory_before["video_count"],
            "total_bytes": inventory_before["total_bytes"],
            "aggregate_sha256": inventory_before["aggregate_sha256"],
        },
    }
    _write_journal(journal_path, journal)
    for name, location in sorted(state.items()):
        if location == "source":
            (storage_root / name).replace(raw_root / name)
            journal["state"][name] = "destination"
            journal["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_journal(journal_path, journal)

    inventory = build_inventory(raw_root)
    write_json(inventory_after_path, inventory)
    expected_fingerprint = (
        inventory_before["video_count"],
        inventory_before["total_bytes"],
        inventory_before["aggregate_sha256"],
    )
    actual_fingerprint = (
        inventory["video_count"],
        inventory["total_bytes"],
        inventory["aggregate_sha256"],
    )
    if actual_fingerprint != expected_fingerprint:
        journal.update(
            {
                "status": "verification_failed",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_journal(journal_path, journal)
        raise ValueError("Post-migration inventory does not match pre-migration inventory")
    journal.update(
        {
            "status": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "video_count": inventory["video_count"],
            "total_bytes": inventory["total_bytes"],
            "post_migration_inventory": {
                "path": str(inventory_after_path),
                "video_count": inventory["video_count"],
                "total_bytes": inventory["total_bytes"],
                "aggregate_sha256": inventory["aggregate_sha256"],
            },
        }
    )
    _write_journal(journal_path, journal)
    return journal


def validate_staged_dataset(candidate_root: Path) -> dict[str, Any]:
    """Validate the filesystem portion of a candidate before indexing."""
    candidate_root = candidate_root.resolve(strict=True)
    processed = candidate_root / "processed"
    manifest_path = processed / "dataset-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Candidate manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "READY":
        raise ValueError("Candidate manifest must have status READY")
    fingerprint = manifest.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", fingerprint
    ) is None:
        raise ValueError("Candidate manifest fingerprint is invalid")
    counts = manifest.get("record_counts")
    expected_videos = counts.get("videos") if isinstance(counts, dict) else None
    if not isinstance(expected_videos, int) or expected_videos < 1:
        raise ValueError("Candidate manifest videos count is invalid")
    actual_videos = len(list((processed / "metadata").glob("*.json")))
    if actual_videos != expected_videos:
        raise ValueError(
            f"Candidate metadata count {actual_videos} != {expected_videos}"
        )
    return manifest


_PUBLISH_COMPONENTS = ("processed", "metadata.db", "databases")


def promote_candidate(
    storage_root: Path,
    candidate_root: Path,
    *,
    dataset_id: str,
) -> dict[str, Any]:
    """Promote same-volume candidate components with a resumable journal."""
    storage_root = storage_root.resolve(strict=True)
    candidate_root = candidate_root.resolve(strict=True)
    if storage_root.anchor.lower() != candidate_root.anchor.lower():
        raise ValueError("Candidate promotion must stay on the same volume")
    missing = [name for name in _PUBLISH_COMPONENTS if not (candidate_root / name).exists()]
    if missing:
        raise ValueError(f"Candidate components are missing: {', '.join(missing)}")

    safe_dataset_id = re.sub(r"[^A-Za-z0-9_.-]", "-", dataset_id)
    rollback_root = storage_root / ".rollback" / safe_dataset_id
    if rollback_root.exists():
        raise ValueError(f"Rollback directory already exists: {rollback_root}")
    rollback_root.mkdir(parents=True)
    journal_path = storage_root / ".migration" / f"publish-{safe_dataset_id}.json"
    journal: dict[str, Any] = {
        "schema_version": 1,
        "status": "promoting",
        "storage_root": str(storage_root),
        "candidate_root": str(candidate_root),
        "rollback_root": str(rollback_root),
        "journal_path": str(journal_path),
        "components": {},
    }
    _write_journal(journal_path, journal)
    try:
        for name in _PUBLISH_COMPONENTS:
            canonical = storage_root / name
            rollback = rollback_root / name
            candidate = candidate_root / name
            had_canonical = canonical.exists()
            if had_canonical:
                canonical.replace(rollback)
            journal["components"][name] = {
                "had_canonical": had_canonical,
                "old_moved": had_canonical,
                "new_moved": False,
            }
            _write_journal(journal_path, journal)
            candidate.replace(canonical)
            journal["components"][name]["new_moved"] = True
            _write_journal(journal_path, journal)
    except Exception:
        rollback_promotion(journal_path)
        raise
    journal["status"] = "promoted"
    _write_journal(journal_path, journal)
    return journal


def rollback_promotion(journal_path: Path) -> dict[str, Any]:
    """Reverse completed publish steps without deleting either dataset."""
    journal_path = journal_path.resolve(strict=True)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    storage_root = Path(journal["storage_root"])
    candidate_root = Path(journal["candidate_root"])
    rollback_root = Path(journal["rollback_root"])
    candidate_root.mkdir(parents=True, exist_ok=True)
    for name in reversed(_PUBLISH_COMPONENTS):
        state = journal.get("components", {}).get(name, {})
        canonical = storage_root / name
        candidate = candidate_root / name
        rollback = rollback_root / name
        if state.get("new_moved") and canonical.exists() and not candidate.exists():
            canonical.replace(candidate)
        if state.get("old_moved") and rollback.exists() and not canonical.exists():
            rollback.replace(canonical)
    journal["status"] = "rolled_back"
    _write_journal(journal_path, journal)
    return journal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--root", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)

    changed = subparsers.add_parser("changed")
    changed.add_argument("--local", type=Path, required=True)
    changed.add_argument("--remote", type=Path)
    changed.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate-stage")
    validate.add_argument("--candidate-root", type=Path, required=True)

    migrate = subparsers.add_parser("migrate-raw")
    migrate.add_argument("--storage-root", type=Path, required=True)
    migrate.add_argument("--expected-batches", type=int, default=14)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--storage-root", type=Path, required=True)
    promote.add_argument("--candidate-root", type=Path, required=True)
    promote.add_argument("--dataset-id", required=True)

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--journal", type=Path, required=True)

    sqlite_ids = subparsers.add_parser("sqlite-video-ids")
    sqlite_ids.add_argument("--database", type=Path, required=True)
    sqlite_ids.add_argument("--output", type=Path, required=True)

    backup = subparsers.add_parser("audit-backup")
    backup.add_argument("--backup-root", type=Path, required=True)
    backup.add_argument("--videos-root", type=Path, required=True)
    backup.add_argument("--expected-videos", type=int, required=True)
    backup.add_argument("--expected-keyframes", type=int, required=True)
    backup.add_argument("--require-objects", action="store_true")
    backup.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "inventory":
        payload = build_inventory(args.root)
        write_json(args.output, payload)
        print(
            f"video_count={payload['video_count']} "
            f"total_bytes={payload['total_bytes']} "
            f"aggregate_sha256={payload['aggregate_sha256']}"
        )
    elif args.command == "changed":
        local = read_json(args.local)
        remote = read_json(args.remote) if args.remote and args.remote.is_file() else None
        files = changed_inventory_files(local, remote)
        payload = {
            "changed_count": len(files),
            "changed_bytes": sum(entry["size_bytes"] for entry in files),
            "files": files,
        }
        write_json(args.output, payload)
        print(
            f"changed_count={payload['changed_count']} "
            f"changed_bytes={payload['changed_bytes']}"
        )
    elif args.command == "validate-stage":
        manifest = validate_staged_dataset(args.candidate_root)
        print(
            f"dataset_id={manifest['dataset_id']} "
            f"dataset_fingerprint={manifest['dataset_fingerprint']}"
        )
    elif args.command == "migrate-raw":
        payload = migrate_raw_layout(
            args.storage_root,
            expected_batch_count=args.expected_batches,
        )
        print(
            f"status={payload['status']} video_count={payload['video_count']} "
            f"total_bytes={payload['total_bytes']}"
        )
    elif args.command == "promote":
        payload = promote_candidate(
            args.storage_root,
            args.candidate_root,
            dataset_id=args.dataset_id,
        )
        print(f"status={payload['status']} journal={payload['journal_path']}")
    elif args.command == "rollback":
        payload = rollback_promotion(args.journal)
        print(f"status={payload['status']} journal={payload['journal_path']}")
    elif args.command == "sqlite-video-ids":
        video_ids = read_sqlite_video_ids(args.database)
        payload = {"video_count": len(video_ids), "video_ids": video_ids}
        write_json(args.output, payload)
        print(f"video_count={len(video_ids)}")
    elif args.command == "audit-backup":
        payload = audit_backup_online_artifacts(
            args.backup_root,
            args.videos_root,
            expected_video_count=args.expected_videos,
            expected_keyframe_count=args.expected_keyframes,
            require_objects=args.require_objects,
        )
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
