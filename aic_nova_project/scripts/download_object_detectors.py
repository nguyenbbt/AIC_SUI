import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


CODETR_CONFIGS = (
    "co_dino_5scale_r50_8xb2_1x_coco",
    "co_dino_5scale_swin_l_16e_o365tococo",
)


def _require_nonempty_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required model artifact is missing: {path}")
    return path


def download_yolo_world(weights_dir: Path = Path("weights")) -> Path:
    """Download YOLO-World and return its validated checkpoint path."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    destination = weights_dir / "yolov8s-world.pt"
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"YOLO-World already exists at {destination}")
        return destination

    print("Downloading YOLO-World (yolov8s-world.pt)...")
    from ultralytics import YOLOWorld

    source = Path("yolov8s-world.pt")
    YOLOWorld(str(source))
    if source.is_file():
        shutil.move(str(source), str(destination))

    return _require_nonempty_file(destination)


def download_codetr(weights_dir: Path = Path("weights")) -> List[Path]:
    """Download and validate both required Co-DETR variants."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[Path] = []

    for config_name in CODETR_CONFIGS:
        print(f"Downloading Co-DETR config: {config_name}")
        subprocess.run(
            [
                "mim",
                "download",
                "mmdet",
                "--config",
                config_name,
                "--dest",
                str(weights_dir),
            ],
            check=True,
        )

        config_files = list(weights_dir.glob(f"{config_name}*.py"))
        checkpoint_files = list(weights_dir.glob(f"{config_name}*.pth"))
        if not config_files or not checkpoint_files:
            raise FileNotFoundError(
                f"Co-DETR download incomplete for {config_name}"
            )

        artifacts.extend(
            _require_nonempty_file(path)
            for path in (config_files[0], checkpoint_files[0])
        )

    return artifacts


def main() -> int:
    try:
        download_yolo_world()
        download_codetr()
    except Exception as exc:
        print(f"Object detector download failed: {exc}", file=sys.stderr)
        return 1

    print("All required object detection artifacts are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
