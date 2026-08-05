import os
from pathlib import Path

from scripts.run_all_tests import SUITE_IMPORT_ROOTS, discover_test_suites


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _repository_test_files() -> set[Path]:
    files: set[Path] = set()
    for current_root, directory_names, file_names in os.walk(PROJECT_ROOT):
        directory_names[:] = [
            name
            for name in directory_names
            if name not in {"venv", ".agents", ".git", "__pycache__"}
        ]
        files.update(
            (Path(current_root) / name).resolve()
            for name in file_names
            if name.startswith("test_") and name.endswith(".py")
        )
    return files


def test_root_test_command_discovers_every_repository_suite() -> None:
    suites = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in discover_test_suites(PROJECT_ROOT)
    }

    assert "feature_extraction/text_embedding/tests" in suites
    assert "indexing/tests" in suites
    assert "tests" in suites
    assert SUITE_IMPORT_ROOTS["feature_extraction/object_detection/tests"] == (
        "feature_extraction/object_detection"
    )
    assert SUITE_IMPORT_ROOTS["feature_extraction/ocr/tests"] == (
        "feature_extraction/ocr/src"
    )

    covered_files = {
        path.resolve()
        for suite in discover_test_suites(PROJECT_ROOT)
        for path in suite.rglob("test_*.py")
    }
    assert covered_files == _repository_test_files()
