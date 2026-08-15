from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class PackageImportTests(unittest.TestCase):
    def test_source_online_package_is_not_shadowed_by_tests_online(self) -> None:
        domain = importlib.import_module("online.domain")
        adapters = importlib.import_module("online.adapters")
        self.assertEqual(Path(domain.__file__).parts[-3:], ("online", "domain", "__init__.py"))
        self.assertEqual(
            Path(adapters.__file__).parts[-3:],
            ("online", "adapters", "__init__.py"),
        )

    def test_retrieval_package_uses_the_source_online_namespace(self) -> None:
        retrieval = importlib.import_module("online.retrieval")
        self.assertEqual(
            Path(retrieval.__file__).parts[-3:],
            ("online", "retrieval", "__init__.py"),
        )

    def test_contract_validator_does_not_require_optional_qwen_http_client(self) -> None:
        script = textwrap.dedent(
            """
            import sys

            class BlockHttpx:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "httpx":
                        raise ModuleNotFoundError("blocked optional httpx dependency")
                    return None

            sys.meta_path.insert(0, BlockHttpx())
            from online.adapters.contract_validator import OfflineContractValidator
            assert OfflineContractValidator is not None
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
