"""Stable, safe error codes at the Online infrastructure boundary."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    INVALID_QUERY = "INVALID_QUERY"
    MISSING_METADATA = "MISSING_METADATA"
    BRANCH_TIMEOUT = "BRANCH_TIMEOUT"


class DataInfrastructureError(RuntimeError):
    code: ErrorCode

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_safe_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "details": self.details}


class ContractMismatchError(DataInfrastructureError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.CONTRACT_MISMATCH, message, details=details)


class ResourceUnavailableError(DataInfrastructureError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.RESOURCE_UNAVAILABLE, message, details=details)


class DimensionMismatchError(DataInfrastructureError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.DIMENSION_MISMATCH, message, details=details)


class InvalidQueryError(DataInfrastructureError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.INVALID_QUERY, message, details=details)


class MissingMetadataError(DataInfrastructureError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.MISSING_METADATA, message, details=details)


class BranchTimeoutError(DataInfrastructureError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.BRANCH_TIMEOUT, message, details=details)
