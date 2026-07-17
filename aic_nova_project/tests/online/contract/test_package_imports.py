from __future__ import annotations

import importlib
import unittest


class PackageImportTests(unittest.TestCase):
    def test_source_online_package_is_not_shadowed_by_tests_online(self) -> None:
        domain = importlib.import_module("online.domain")
        adapters = importlib.import_module("online.adapters")
        self.assertTrue(domain.__file__.endswith("online\\domain\\__init__.py"))
        self.assertTrue(adapters.__file__.endswith("online\\adapters\\__init__.py"))
