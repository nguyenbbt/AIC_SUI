"""Run every repository pytest suite in an isolated Python process."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".agents",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "venv",
}
SUITE_IMPORT_ROOTS = {
    "feature_extraction/object_detection/tests": (
        "feature_extraction/object_detection"
    ),
    "feature_extraction/ocr/tests": "feature_extraction/ocr/src",
    "feature_extraction/text_embedding/tests": (
        "feature_extraction/text_embedding"
    ),
    "indexing/tests": "indexing",
}


def discover_test_suites(project_root: Path) -> list[Path]:
    """Return every directory named ``tests`` containing pytest files."""
    suites: list[Path] = []
    for current_root, directory_names, file_names in os.walk(project_root):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in EXCLUDED_DIRECTORIES
        )
        current_path = Path(current_root)
        if current_path.name != "tests":
            continue
        if any(
            name.startswith("test_") and name.endswith(".py")
            for name in file_names
        ):
            suites.append(current_path)
        # One pytest invocation at this level also collects nested test files.
        directory_names.clear()

    return sorted(suites, key=lambda path: path.as_posix())


def run_test_suites(
    project_root: Path,
    suites: Sequence[Path],
    pytest_args: Sequence[str],
) -> int:
    """Run suites independently to avoid collisions between ``src`` layouts."""
    failed_suites: list[Path] = []
    for suite in suites:
        relative_suite = suite.relative_to(project_root)
        relative_key = relative_suite.as_posix()
        import_root_value = SUITE_IMPORT_ROOTS.get(relative_key)
        if import_root_value is None:
            working_directory = project_root
            pytest_target = relative_suite
            environment = None
        else:
            import_root = project_root / import_root_value
            working_directory = import_root
            if import_root.name == "src":
                working_directory = import_root.parent
            pytest_target = suite.relative_to(working_directory)
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (str(import_root), existing_pythonpath)
                if part
            )

        print(f"\n=== pytest {relative_suite.as_posix()} ===", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(pytest_target),
                *pytest_args,
            ],
            cwd=working_directory,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            failed_suites.append(relative_suite)

    if failed_suites:
        print("\nFailed test suites:", file=sys.stderr)
        for suite in failed_suites:
            print(f"- {suite.as_posix()}", file=sys.stderr)
        return 1

    print(f"\nAll {len(suites)} test suites passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the repository-wide test command."""
    parser = argparse.ArgumentParser(
        description="Discover and run all repository pytest suites."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered suites without running pytest.",
    )
    options, pytest_args = parser.parse_known_args(argv)
    suites = discover_test_suites(PROJECT_ROOT)

    if options.list:
        for suite in suites:
            print(suite.relative_to(PROJECT_ROOT).as_posix())
        return 0
    if not suites:
        print("No pytest suites were discovered.", file=sys.stderr)
        return 1

    return run_test_suites(PROJECT_ROOT, suites, pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
