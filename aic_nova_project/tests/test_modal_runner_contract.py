import ast
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODAL_RUNNER = PROJECT_ROOT / "scripts" / "offline_modal_runner.py"


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
