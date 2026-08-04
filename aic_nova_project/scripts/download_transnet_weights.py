import shutil
import sys
from pathlib import Path


DEFAULT_WEIGHTS_PATH = Path("weights/transnetv2-pytorch-weights.pth")


def _require_nonempty_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required TransNetV2 weights are missing: {path}")
    return path


def download_weights(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> Path:
    """Copy packaged TransNetV2 weights and validate the destination."""
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if weights_path.is_file() and weights_path.stat().st_size > 0:
        print(f"Weights already exist at {weights_path}")
        return weights_path

    import transnetv2_pytorch

    package_dir = Path(transnetv2_pytorch.__file__).resolve().parent
    package_weights = package_dir / "transnetv2-pytorch-weights.pth"
    _require_nonempty_file(package_weights)

    print(f"Copying packaged weights from {package_weights} to {weights_path}")
    shutil.copy2(package_weights, weights_path)
    return _require_nonempty_file(weights_path)


def main() -> int:
    try:
        download_weights()
    except Exception as exc:
        print(f"TransNetV2 weights extraction failed: {exc}", file=sys.stderr)
        return 1

    print("TransNetV2 weights are ready for offline usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
