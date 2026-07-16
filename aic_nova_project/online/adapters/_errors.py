"""Backend exception translation shared by adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from online.domain.errors import BranchTimeoutError, DataInfrastructureError, ResourceUnavailableError


T = TypeVar("T")


def call_backend(operation: str, resource: str, function: Callable[[], T]) -> T:
    try:
        return function()
    except DataInfrastructureError:
        raise
    except Exception as exc:
        details = {"operation": operation, "resource": resource}
        if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
            raise BranchTimeoutError(
                f"Backend timed out during {operation}", details=details
            ) from exc
        raise ResourceUnavailableError(
            f"Backend unavailable during {operation}", details=details
        ) from exc
