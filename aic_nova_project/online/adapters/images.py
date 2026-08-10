"""Filesystem-backed, traversal-safe keyframe resolver for VQA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from online.domain.errors import ContractMismatchError, InvalidQueryError, ResourceUnavailableError
from online.domain.vqa import ImageEvidence
from online.ports.metadata import MetadataReaderPort


class FilesystemImageResolver:
    """Verify Offline image artifacts and expose only dataset-relative references."""

    def __init__(self, *, data_root: Path, metadata_reader: MetadataReaderPort) -> None:
        if not isinstance(metadata_reader, MetadataReaderPort):
            raise TypeError("metadata_reader must implement MetadataReaderPort")
        self._data_root = Path(data_root).expanduser().resolve()
        self._metadata_reader = metadata_reader

    def health_check(self) -> None:
        if not self._data_root.is_dir():
            raise ResourceUnavailableError(
                "Configured dataset root does not exist",
                details={"resource": "data_root"},
            )

    def resolve_images(self, frame_ids: Sequence[str]) -> Mapping[str, ImageEvidence]:
        ids = _validate_ids(frame_ids)
        if not ids:
            return {}
        metadata = self._metadata_reader.get_frames_by_ids(ids)
        if not isinstance(metadata, Mapping):
            raise ContractMismatchError("Metadata reader returned a non-mapping value")

        output: dict[str, ImageEvidence] = {}
        for frame_id in ids:
            frame = metadata.get(frame_id)
            if frame is None:
                continue
            candidate = (self._data_root / Path(*frame.image_rel_path.split("/"))).resolve()
            if not candidate.is_relative_to(self._data_root):
                raise ContractMismatchError("Keyframe path escapes the configured dataset root")
            # Missing/unreadable images are deliberately omitted so VQA can report
            # bounded missing-evidence diagnostics instead of fabricating evidence.
            if not candidate.is_file():
                continue
            try:
                with candidate.open("rb") as stream:
                    if not stream.read(1):
                        continue
            except OSError:
                continue
            output[frame_id] = ImageEvidence(
                evidence_id=f"image:{frame.frame_id}",
                video_id=frame.video_id,
                frame_id=frame.frame_id,
                shot_id=frame.shot_id,
                timestamp_sec=frame.timestamp_sec,
                source_frame_idx=frame.source_frame_idx,
                image_reference=frame.image_rel_path,
            )
        return output


def _validate_ids(frame_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(frame_ids, (str, bytes)):
        raise InvalidQueryError("frame_ids must be a sequence of strings")
    try:
        values = tuple(dict.fromkeys(frame_ids))
    except (TypeError, ValueError) as exc:
        raise InvalidQueryError("frame_ids must be a sequence of strings") from exc
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in values
    ):
        raise InvalidQueryError("frame_ids must contain canonical non-empty strings")
    return values


__all__ = ["FilesystemImageResolver"]
