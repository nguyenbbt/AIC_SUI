"""Portable Modal entrypoint for Offline Modules 1-7.

Examples::

    modal run modal_runner.py --module module1 \
        --arguments="--input /data/raw_videos --output /data/processed"
    modal run modal_runner.py --module module6 \
        --arguments="--asr-dir /data/processed/transcripts \
        --summary-dir /data/processed/summaries \
        --ocr-dir /data/processed/ocr \
        --output-dir /data/processed/embeddings"

Select the Modal Volume with ``AIC_MODAL_DATA_VOLUME`` before a run.
All module arguments are forwarded without invoking a shell.
"""

from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import sys
from typing import Sequence

import modal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPOSITORY_ROOT = "/workspace"
REMOTE_DATA_ROOT = "/data"
NUMPY_BINARY_REQUIREMENT = "numpy==1.26.4"
OPENCV_BINARY_REQUIREMENT = "opencv-python-headless==4.9.0.80"
SETUPTOOLS_RUNTIME_REQUIREMENT = "setuptools==81.0.0"
DATA_VOLUME_NAME = os.environ.get(
    "AIC_MODAL_DATA_VOLUME",
    "aic-nova-offline-data",
)

OFFLINE_MODULES = {
    "module1": {
        "entrypoint": "data_pipeline.shot_keyframe.cli",
        "cwd": "/workspace",
        "pythonpath": "/workspace",
    },
    "module2": {
        "entrypoint": "feature_extraction.visual_embedding.cli",
        "cwd": "/workspace",
        "pythonpath": "/workspace",
    },
    "module3": {
        "entrypoint": "feature_extraction.asr_transcript.cli",
        "cwd": "/workspace",
        "pythonpath": "/workspace",
    },
    "module4": {
        "entrypoint": "ocr_module.cli",
        "cwd": "/workspace/feature_extraction/ocr",
        "pythonpath": "/workspace/feature_extraction/ocr/src",
    },
    "module5": {
        "entrypoint": "src.object_detection.cli",
        "cwd": "/workspace/feature_extraction/object_detection",
        "pythonpath": "/workspace/feature_extraction/object_detection",
    },
    "module6": {
        "entrypoint": "src.text_embedding.cli",
        "cwd": "/workspace/feature_extraction/text_embedding",
        "pythonpath": "/workspace/feature_extraction/text_embedding",
    },
    "module7": {
        "entrypoint": "src.indexing.cli",
        "cwd": "/workspace/indexing",
        "pythonpath": "/workspace/indexing",
    },
}

REQUIREMENT_FILES = (
    "data_pipeline/shot_keyframe/requirements.txt",
    "feature_extraction/visual_embedding/requirements.txt",
    "feature_extraction/asr_transcript/requirements.txt",
    "feature_extraction/ocr/requirements.txt",
    "feature_extraction/object_detection/requirements.txt",
    "feature_extraction/text_embedding/requirements.txt",
    "indexing/requirements.txt",
)


def _build_image() -> modal.Image:
    """Build one reproducible image containing all Offline dependencies."""
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])
        .apt_install(
            "build-essential",
            "ca-certificates",
            "ffmpeg",
            "git",
            "libgl1",
            "libglib2.0-0",
        )
    )
    for relative_path in REQUIREMENT_FILES:
        image = image.pip_install_from_requirements(
            REPOSITORY_ROOT / relative_path
        )
    image = (
        image.pip_install(
            NUMPY_BINARY_REQUIREMENT,
            OPENCV_BINARY_REQUIREMENT,
            SETUPTOOLS_RUNTIME_REQUIREMENT,
        ).run_commands(
            "python -c \"import cv2, gdown, numpy, pkg_resources; "
            "assert numpy.__version__ == '1.26.4', numpy.__version__; "
            "print(numpy.__version__, cv2.__version__)\""
        )
    )
    return image.add_local_dir(
        REPOSITORY_ROOT,
        remote_path=REMOTE_REPOSITORY_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".agents",
            ".pytest_cache",
            "__pycache__",
            "venv",
            "node_modules",
        ],
    )


app = modal.App("aic-nova-offline")
offline_image = _build_image()
data_volume = modal.Volume.from_name(
    DATA_VOLUME_NAME,
    create_if_missing=True,
)


def build_volume_inventory(root: Path) -> dict[str, int | str]:
    """Hash a remote video tree using the local inventory contract."""
    extensions = {".mp4", ".mkv", ".avi", ".webm"}
    entries: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            (path.relative_to(root).as_posix(), path.stat().st_size, digest.hexdigest())
        )
    aggregate = hashlib.sha256()
    for relative_path, size_bytes, digest in entries:
        aggregate.update(
            f"{relative_path}\0{size_bytes}\0{digest}\n".encode("utf-8")
        )
    return {
        "video_count": len(entries),
        "total_bytes": sum(entry[1] for entry in entries),
        "aggregate_sha256": aggregate.hexdigest(),
    }


@app.function(
    image=offline_image,
    timeout=86_400,
    volumes={REMOTE_DATA_ROOT: data_volume},
)
def verify_volume_inventory(root: str) -> dict[str, int | str]:
    """Compute a CPU-only integrity summary before any paid GPU module."""
    data_volume.reload()
    path = Path(root).resolve()
    data_root = Path(REMOTE_DATA_ROOT).resolve()
    if path != data_root and data_root not in path.parents:
        raise ValueError(f"Inventory root must stay under {REMOTE_DATA_ROOT}")
    if not path.is_dir():
        raise FileNotFoundError(f"Inventory root not found: {path}")
    return build_volume_inventory(path)


def build_module_command(
    module_name: str,
    arguments: Sequence[str],
) -> tuple[list[str], str, dict[str, str]]:
    """Resolve one allowlisted module into a shell-free subprocess call."""
    try:
        config = OFFLINE_MODULES[module_name]
    except KeyError as exc:
        choices = ", ".join(sorted(OFFLINE_MODULES))
        raise ValueError(
            f"Unknown Offline module '{module_name}'. Choose one of: {choices}."
        ) from exc

    environment = os.environ.copy()
    environment["PYTHONPATH"] = config["pythonpath"]
    command = [
        sys.executable,
        "-m",
        config["entrypoint"],
        *arguments,
    ]
    return command, config["cwd"], environment


@app.function(
    image=offline_image,
    gpu="A10G",
    timeout=86_400,
    volumes={REMOTE_DATA_ROOT: data_volume},
)
def run_offline_module(module_name: str, arguments: list[str]) -> None:
    """Run one Offline module and commit outputs only after success."""
    command, working_directory, environment = build_module_command(
        module_name,
        arguments,
    )
    print(f"Running {module_name}: {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        check=True,
    )
    data_volume.commit()


@app.local_entrypoint()
def main(
    module: str = "",
    arguments: str = "",
    verify_root: str = "",
    expected_count: int = -1,
    expected_bytes: int = -1,
    expected_digest: str = "",
) -> None:
    """Parse a quoted module argument string and launch it remotely."""
    if verify_root:
        actual = verify_volume_inventory.remote(verify_root)
        expected = {
            "video_count": expected_count,
            "total_bytes": expected_bytes,
            "aggregate_sha256": expected_digest,
        }
        if actual != expected:
            raise RuntimeError(
                "Remote inventory mismatch: "
                f"expected={json.dumps(expected, sort_keys=True)} "
                f"actual={json.dumps(actual, sort_keys=True)}"
            )
        print(json.dumps(actual, sort_keys=True))
        return
    if module not in OFFLINE_MODULES:
        choices = ", ".join(sorted(OFFLINE_MODULES))
        raise ValueError(
            f"Unknown Offline module '{module}'. Choose one of: {choices}."
        )
    run_offline_module.remote(module, shlex.split(arguments))
