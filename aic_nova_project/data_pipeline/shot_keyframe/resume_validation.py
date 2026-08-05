from pathlib import Path

from PIL import Image

from .metadata_schema import VideoMetadata


def keyframe_artifacts_are_valid(
    metadata: VideoMetadata,
    output_dir: str | Path,
) -> bool:
    """Return whether every keyframe exists and is a readable image."""
    output_root = Path(output_dir)

    for shot in metadata.shots:
        for keyframe in shot.keyframes:
            image_path = output_root / keyframe.file_path
            if not image_path.is_file():
                return False

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError):
                return False

    return True
