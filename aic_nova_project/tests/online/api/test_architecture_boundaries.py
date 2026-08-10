from __future__ import annotations

import ast
from pathlib import Path


def test_production_modules_do_not_import_testing_packages() -> None:
    project_root = Path(__file__).resolve().parents[3]
    violations: list[str] = []
    for source_root in (project_root / "online", project_root / "retrieval_api"):
        for path in source_root.rglob("*.py"):
            if source_root.name == "online" and "testing" in path.relative_to(source_root).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    if (
                        module == "tests"
                        or module.startswith("tests.")
                        or module == "online.testing"
                        or module.startswith("online.testing.")
                    ):
                        violations.append(f"{path.relative_to(project_root)} imports {module}")
    assert not violations, "\n".join(violations)
