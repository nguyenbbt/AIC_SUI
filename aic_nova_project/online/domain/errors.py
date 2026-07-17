"""Stable, safe error codes at the Online infrastructure boundary."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


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
        self.details = MappingProxyType(_sanitize_details(details or {}))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }


_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "token",
    "secret",
    "authorization",
    "credential",
    "vector",
    "embedding",
)


def _redact_uri(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[redacted]"
    if not parsed.scheme or parsed.hostname is None:
        return value
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def _sanitize_details(details: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded, credential-free diagnostic details for public boundaries."""

    output: dict[str, Any] = {}
    for raw_key, value in details.items():
        key = str(raw_key)
        lowered = key.lower()
        if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            output[key] = "[redacted]"
        elif "uri" in lowered and isinstance(value, str):
            output[key] = _redact_uri(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value if not isinstance(value, str) or len(value) <= 500 else value[:497] + "..."
        elif isinstance(value, Enum):
            output[key] = value.value
        elif isinstance(value, (tuple, list, set)):
            output[key] = f"[{type(value).__name__} length={len(value)}]"
        elif isinstance(value, Mapping):
            output[key] = f"[mapping keys={len(value)}]"
        else:
            output[key] = f"[{type(value).__name__}]"
    return output


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
