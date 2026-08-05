import ast
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODAL_RUNNER = PROJECT_ROOT / "scripts" / "offline_modal_runner.py"
NUMPY_REQUIREMENT = "numpy==1.26.4"
OPENCV_REQUIREMENT = "opencv-python-headless==4.9.0.80"
SETUPTOOLS_REQUIREMENT = "setuptools==81.0.0"


def _offline_modules(source: str) -> dict:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == "OFFLINE_MODULES"
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError("OFFLINE_MODULES registry is missing")


def test_modal_runner_is_portable_and_registers_modules_1_to_7() -> None:
    source = MODAL_RUNNER.read_text(encoding="utf-8")

    assert not re.search(r"[A-Za-z]:[/\\]", source)
    assert "app.main:app" not in source
    assert "shell=True" not in source
    assert set(_offline_modules(source)) == {
        "module1",
        "module2",
        "module3",
        "module4",
        "module5",
        "module6",
        "module7",
    }


def test_modal_requirements_use_one_numpy_opencv_binary_contract() -> None:
    requirement_paths = [
        PROJECT_ROOT / "data_pipeline" / "shot_keyframe" / "requirements.txt",
        PROJECT_ROOT
        / "feature_extraction"
        / "visual_embedding"
        / "requirements.txt",
        PROJECT_ROOT / "feature_extraction" / "ocr" / "requirements.txt",
    ]
    requirements = [
        line.strip()
        for path in requirement_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    numpy_requirements = [
        item for item in requirements if item.lower().startswith("numpy")
    ]
    opencv_requirements = [
        item for item in requirements if item.lower().startswith("opencv-")
    ]

    assert set(numpy_requirements) == {NUMPY_REQUIREMENT}
    assert set(opencv_requirements) == {OPENCV_REQUIREMENT}


def test_modal_image_reasserts_and_import_checks_binary_dependencies() -> None:
    source = MODAL_RUNNER.read_text(encoding="utf-8")
    ocr_requirements = (
        PROJECT_ROOT / "feature_extraction" / "ocr" / "requirements.txt"
    ).read_text(encoding="utf-8")

    assert f'"{NUMPY_REQUIREMENT}"' in source
    assert f'"{OPENCV_REQUIREMENT}"' in source
    assert f'"{SETUPTOOLS_REQUIREMENT}"' in source
    assert SETUPTOOLS_REQUIREMENT in ocr_requirements
    assert "import cv2, gdown, numpy, pkg_resources" in source
