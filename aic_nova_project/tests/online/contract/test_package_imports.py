from __future__ import annotations

import importlib
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
