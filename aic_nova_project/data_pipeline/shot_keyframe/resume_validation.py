from pathlib import Path

from PIL import Image

from .fingerprints import sha256_file
from .metadata_schema import VideoMetadata


def keyframe_artifacts_are_valid(
    metadata: VideoMetadata,
    output_dir: str | Path,
    *,
    expected_source_fingerprint: str | None = None,
    expected_config_fingerprint: str | None = None,
    expected_source_video_rel_path: str | None = None,
) -> bool:
    """Return whether provenance and every published keyframe still match."""
    output_root = Path(output_dir)

    if (
        expected_source_fingerprint is not None
        and metadata.source_fingerprint != expected_source_fingerprint
    ):
        return False
    if (
        expected_config_fingerprint is not None
        and metadata.producer_config_fingerprint
        != expected_config_fingerprint
    ):
        return False
    if (
        expected_source_video_rel_path is not None
        and metadata.source_video_rel_path
        != expected_source_video_rel_path
    ):
        return False

    for shot in metadata.shots:
        for keyframe in shot.keyframes:
            image_path = output_root / keyframe.file_path
            if not image_path.is_file() or keyframe.image_sha256 is None:
                return False

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError):
                return False
            try:
                image_fingerprint = sha256_file(image_path)
            except OSError:
                return False
            if image_fingerprint != keyframe.image_sha256:
                return False

    return True
