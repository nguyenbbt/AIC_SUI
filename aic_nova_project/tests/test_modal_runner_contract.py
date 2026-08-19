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


def test_embedding_model_runtimes_are_exactly_pinned() -> None:
    visual_requirements = (
        PROJECT_ROOT
        / "feature_extraction"
        / "visual_embedding"
        / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()
    text_requirements = (
        PROJECT_ROOT
        / "feature_extraction"
        / "text_embedding"
        / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert {
        "torch==2.13.0",
        "torchvision==0.28.0",
        "Pillow==12.3.0",
        "open_clip_torch==3.3.0",
    } <= set(visual_requirements)
    assert {
        "torch==2.13.0",
        "sentence-transformers==5.6.0",
        "transformers==5.13.1",
        "tokenizers==0.22.2",
        "pandas==3.0.3",
        "pyarrow==25.0.0",
    } <= set(text_requirements)


def test_indexing_verifier_dependencies_are_runtime_pinned() -> None:
    indexing_requirements = (
        PROJECT_ROOT / "indexing" / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert {
        "Pillow==12.3.0",
        "pydantic==2.13.4",
        "pymilvus>=2.4.0,<3.0.0",
    } <= set(indexing_requirements)


def test_online_encoder_runtime_matches_offline_model_stack() -> None:
    online_requirements = (
        PROJECT_ROOT / "online" / "requirements-encoders.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert {
        "torch==2.13.0",
        "torchvision==0.28.0",
        "open_clip_torch==3.3.0",
        "sentence-transformers==5.6.0",
        "transformers==5.13.1",
        "tokenizers==0.22.2",
    } <= set(online_requirements)


def test_indexing_image_contains_online_contract_validator() -> None:
    dockerfile = (PROJECT_ROOT / "indexing" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY online/ /app/online/" in dockerfile


def test_modal_runner_has_cpu_only_remote_inventory_gate() -> None:
    source = MODAL_RUNNER.read_text(encoding="utf-8")

    assert "def build_volume_inventory" in source
    assert "def verify_volume_inventory" in source
    assert "expected_digest" in source
    verifier_prefix = source.split("def verify_volume_inventory", 1)[0]
    verifier_decorator = verifier_prefix.rsplit("@app.function", 1)[1]
    assert "volumes={REMOTE_DATA_ROOT: data_volume}" in verifier_decorator
    assert "gpu=" not in verifier_decorator


def test_modal_runner_volume_name_is_selected_by_process_environment() -> None:
    source = MODAL_RUNNER.read_text(encoding="utf-8")
    compact = " ".join(source.split())

    assert 'os.environ.get( "AIC_MODAL_DATA_VOLUME"' in compact
    assert "modal.Volume.from_name( DATA_VOLUME_NAME" in compact


def test_modal_runner_parallelizes_module3_across_gpu_containers() -> None:
    source = MODAL_RUNNER.read_text(encoding="utf-8")

    assert "module3_shards" in source
    assert "run_offline_module.starmap" in source
    assert '"--shard-count"' in source
    assert '"--shard-index"' in source
    assert "@modal.concurrent" not in source
    assert "max_containers=5" in source


def test_modal_runner_parallelizes_module4_across_gpu_containers() -> None:
    source = MODAL_RUNNER.read_text(encoding="utf-8")

    assert "module4_shards" in source
    assert "run_offline_module.starmap" in source
    assert '"--shard-count"' in source
    assert '"--shard-index"' in source
    assert "max_containers=5" in source


def test_modal_runner_has_opt_in_gpu_concurrency_probe() -> None:
    source = MODAL_RUNNER.read_text(encoding="utf-8")

    assert "def probe_gpu_slot" in source
    assert "probe_gpu_slot.starmap" in source
    assert "probe_gpus" in source
    assert "observed_peak_gpu_concurrency" in source
