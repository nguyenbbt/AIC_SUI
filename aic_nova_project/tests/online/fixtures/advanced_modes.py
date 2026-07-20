"""Test access to the single source-of-truth advanced-mode fixture."""

from __future__ import annotations

from online.testing import AdvancedModesFixture, build_advanced_modes_fixture


def advanced_modes_fixture() -> AdvancedModesFixture:
    """Return a fresh deterministic fixture without requiring a test framework."""

    return build_advanced_modes_fixture()


__all__ = ["advanced_modes_fixture"]
