"""Deterministic Wave 3 runtime fakes and lifecycle controls.

The objects in this module deliberately wrap the Wave 2 fakes.  They add a
single public composition bundle, safe cross-resource call logging, injectable
runtime states, and lifecycle gates without changing the frozen domain or port
contracts.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from time import monotonic_ns
from types import MappingProxyType
from typing import Any, TypeVar

from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DataInfrastructureError,
    DimensionMismatchError,
    InvalidQueryError,
    MissingMetadataError,
    ResourceUnavailableError,
)
from online.domain.vqa import (
    ASREvidence,
    ImageEvidence,
    OCREvidence,
    SummaryEvidence,
    VLMConfidence,
    VLMRequest,
    VLMResponse,
    VLMResponseStatus,
)
from online.ports.encoders import TextEncoderPort
from online.ports.evidence import EvidenceHydrationPort
from online.ports.images import ImageResolverPort
from online.ports.metadata import MetadataReaderPort
from online.ports.records import FrameMetadata, VideoMetadata
from online.ports.visual_corpus import (
    OrderedVisualFrame,
    VisualCorpusPort,
    validate_ordered_visual_stream,
)
from online.ports.vlm import VLMPort

from .advanced_fakes import AdvancedModesFixture, build_advanced_modes_fixture


_T = TypeVar("_T")
_MAX_LOG_IDENTIFIERS = 16
_MAX_LOG_IDENTIFIER_LENGTH = 160
_SENSITIVE_LOG_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
    "signature",
    "token",
)


class AdvancedRuntimeState(str, Enum):
    """One deterministic state injected at a fake resource boundary."""

    SUCCESS = "success"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_REFERENCE = "invalid_reference"
    INVALID_REF = "invalid_reference"


# Short aliases keep handoff code readable without creating a second contract.
FakeRuntimeState = AdvancedRuntimeState


def _validated_request_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("request_id must be a bounded normalized string")
    return value


def _coerce_state(value: AdvancedRuntimeState | str, *, field_name: str) -> AdvancedRuntimeState:
    try:
        return AdvancedRuntimeState(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid AdvancedRuntimeState") from exc


@dataclass(frozen=True, slots=True)
class AdvancedRuntimeConfig:
    """Immutable behavior and blocking configuration for one runtime bundle."""

    request_id: str = field(default="advanced-wave3-happy", repr=False)
    encoder_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    visual_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    metadata_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    ocr_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    asr_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    summary_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    image_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    vlm_state: AdvancedRuntimeState = AdvancedRuntimeState.SUCCESS
    block_trake: bool = False
    block_vqa: bool = False
    block_timeout_sec: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _validated_request_id(self.request_id))
        for field_name in (
            "encoder_state",
            "visual_state",
            "metadata_state",
            "ocr_state",
            "asr_state",
            "summary_state",
            "image_state",
            "vlm_state",
        ):
            object.__setattr__(
                self,
                field_name,
                _coerce_state(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("block_trake", "block_vqa"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if (
            isinstance(self.block_timeout_sec, bool)
            or not isinstance(self.block_timeout_sec, (int, float))
            or not math.isfinite(float(self.block_timeout_sec))
            or float(self.block_timeout_sec) <= 0.0
            or float(self.block_timeout_sec) > 60.0
        ):
            raise ValueError("block_timeout_sec must be within (0, 60]")
        object.__setattr__(self, "block_timeout_sec", float(self.block_timeout_sec))

    def for_request(self, request_id: str) -> "AdvancedRuntimeConfig":
        """Return a fresh immutable config with a different safe request ID."""

        return replace(self, request_id=request_id)


@dataclass(frozen=True, slots=True)
class AdvancedRuntimeCall:
    """Bounded SDK-neutral observation of one fake boundary call."""

    sequence: int
    request_id: str
    component: str
    method: str
    state: AdvancedRuntimeState
    item_count: int = 0
    identifiers: tuple[str, ...] = ()

    @property
    def order(self) -> int:
        return self.sequence

    @property
    def operation(self) -> str:
        return self.method


# Alternate name retained as a discoverable handoff convenience.
RuntimeCall = AdvancedRuntimeCall


def _safe_log_identifier(value: object) -> str:
    if not isinstance(value, str):
        return "[invalid-id]"
    lowered = value.lower()
    if (
        not value
        or len(value) > _MAX_LOG_IDENTIFIER_LENGTH
        or any(part in lowered for part in _SENSITIVE_LOG_PARTS)
        or any(
            marker in value
            for marker in ("\\", "/", "?", "#", "=", "file://", "http://", "https://")
        )
        or any(ord(character) < 32 for character in value)
    ):
        return "[redacted-id]"
    return value


def _safe_log_request_id(value: object) -> str:
    """Keep normal opaque request IDs correlated while redacting payload-like IDs."""

    if not isinstance(value, str):
        return "[invalid-request-id]"
    lowered = value.lower()
    windows_absolute = (
        len(value) >= 3
        and value[0].isalpha()
        and value[1] == ":"
        and value[2] in ("\\", "/")
    )
    if (
        not value
        or len(value) > _MAX_LOG_IDENTIFIER_LENGTH
        or any(part in lowered for part in _SENSITIVE_LOG_PARTS)
        or value.startswith(("/", "\\"))
        or windows_absolute
        or "://" in value
        or any(marker in value for marker in ("\\", "?", "#", "="))
        or any(ord(character) < 32 for character in value)
    ):
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"[redacted-request-id:{digest}]"
    return value


class AdvancedRuntimeCallLog:
    """Thread-safe ordered log whose snapshots cannot mutate internal state."""

    def __init__(self) -> None:
        self._calls: list[AdvancedRuntimeCall] = []
        self._next_sequence = 1
        self._lock = threading.Lock()

    def record(
        self,
        *,
        request_id: str,
        component: str,
        method: str,
        state: AdvancedRuntimeState,
        item_count: int = 0,
        identifiers: Sequence[str] = (),
    ) -> AdvancedRuntimeCall:
        if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count < 0:
            raise ValueError("item_count must be a non-negative integer")
        bounded_identifiers = tuple(
            _safe_log_identifier(value)
            for value in tuple(identifiers)[:_MAX_LOG_IDENTIFIERS]
        )
        call = AdvancedRuntimeCall(
            sequence=0,
            request_id=_safe_log_request_id(request_id),
            component=_safe_log_identifier(component),
            method=_safe_log_identifier(method),
            state=state,
            item_count=item_count,
            identifiers=bounded_identifiers,
        )
        with self._lock:
            call = replace(call, sequence=self._next_sequence)
            self._next_sequence += 1
            self._calls.append(call)
        return call

    @property
    def calls(self) -> tuple[AdvancedRuntimeCall, ...]:
        return self.snapshot()

    def snapshot(self) -> tuple[AdvancedRuntimeCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def for_request(self, request_id: str) -> tuple[AdvancedRuntimeCall, ...]:
        safe_request_id = _safe_log_request_id(request_id)
        return tuple(
            call for call in self.snapshot() if call.request_id == safe_request_id
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._calls)

    def __iter__(self) -> Iterator[AdvancedRuntimeCall]:
        return iter(self.snapshot())

    def __getitem__(self, index: int) -> AdvancedRuntimeCall:
        return self.snapshot()[index]


class LifecycleEvent(asyncio.Event):
    """An ``asyncio.Event`` that can also safely gate executor worker threads."""

    def __init__(self) -> None:
        super().__init__()
        self._thread_event = threading.Event()
        self._event_lock = threading.Lock()
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    async def wait(self) -> bool:
        loop = asyncio.get_running_loop()
        with self._event_lock:
            if self._bound_loop is None:
                self._bound_loop = loop
            elif self._bound_loop is not loop and not self._thread_event.is_set():
                raise RuntimeError("LifecycleEvent cannot wait on multiple event loops")
        if self._thread_event.is_set() and not super().is_set():
            super().set()
        return await super().wait()

    def set(self) -> None:
        self._thread_event.set()
        with self._event_lock:
            loop = self._bound_loop
        if loop is None or not loop.is_running():
            super().set()
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            super().set()
        else:
            loop.call_soon_threadsafe(self._set_async_value)

    def _set_async_value(self) -> None:
        super().set()

    def clear(self) -> None:
        self._thread_event.clear()
        with self._event_lock:
            self._bound_loop = None
        # asyncio's LoopBoundMixin retains the first loop independently from
        # Event.clear().  The fake is intentionally reusable across isolated
        # test loops after all prior waiters have observed a set event.
        self._loop = None
        super().clear()

    def is_set(self) -> bool:
        return self._thread_event.is_set()

    def wait_blocking(self, timeout: float | None = None) -> bool:
        return self._thread_event.wait(timeout)


def _immutable_test_value(value: object) -> object:
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _immutable_test_value(item)
                for key, item in dict(value).items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_test_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_test_value(item) for item in value)
    return value


class BlockingLifecycleFake:
    """Reusable active-call gate for graceful-shutdown composition tests.

    ``execute`` covers lifecycle-only async tests.  Runtime port wrappers use
    ``operation`` so the same public started/release events also work when the
    port is called from a thread executor.
    """

    def __init__(
        self,
        name: str = "advanced_lifecycle",
        *,
        timeout_sec: float = 10.0,
        return_value: object = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("lifecycle fake name must be a normalized string")
        if _safe_log_identifier(name) != name:
            raise ValueError("lifecycle fake name must be a safe opaque identifier")
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(float(timeout_sec))
            or float(timeout_sec) <= 0.0
            or float(timeout_sec) > 60.0
        ):
            raise ValueError("timeout_sec must be within (0, 60]")
        self.name = name
        self.timeout_sec = float(timeout_sec)
        self._return_value = _immutable_test_value(return_value)
        self.started_event = LifecycleEvent()
        self.release_event = LifecycleEvent()
        self.close_called_event = LifecycleEvent()
        # Short aliases match normal asyncio.Event test idioms.
        self.started = self.started_event
        self.release = self.release_event
        self.close_called = self.close_called_event
        self.request_started_event = self.started_event
        self.request_release_event = self.release_event
        self._condition = threading.Condition(threading.Lock())
        self._active_count = 0
        self._closed = False
        self._close_count = 0
        self._close_attempt_count = 0
        self._close_attempted_while_active = False
        self._close_prepared = False
        self._first_close_attempt_at_ns: int | None = None
        self._closed_at_ns: int | None = None
        self._last_completion_at_ns: int | None = None
        self._call_log = AdvancedRuntimeCallLog()

    @property
    def calls(self) -> tuple[AdvancedRuntimeCall, ...]:
        return self._call_log.snapshot()

    @property
    def return_value(self) -> object:
        return self._return_value

    @property
    def call_log(self) -> AdvancedRuntimeCallLog:
        return self._call_log

    @property
    def active_count(self) -> int:
        with self._condition:
            return self._active_count

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def close_count(self) -> int:
        with self._condition:
            return self._close_count

    @property
    def close_attempt_count(self) -> int:
        with self._condition:
            return self._close_attempt_count

    @property
    def release_count(self) -> int:
        return int(self.release_event.is_set())

    @property
    def close_attempted_while_active(self) -> bool:
        with self._condition:
            return self._close_attempted_while_active

    @property
    def closed_before_release(self) -> bool:
        return self.close_attempted_while_active

    @property
    def first_close_attempt_at_ns(self) -> int | None:
        with self._condition:
            return self._first_close_attempt_at_ns

    @property
    def close_called_at_ns(self) -> int | None:
        return self.first_close_attempt_at_ns

    @property
    def close_time_ns(self) -> int | None:
        return self.closed_at_ns

    @property
    def closed_at_ns(self) -> int | None:
        with self._condition:
            return self._closed_at_ns

    @property
    def last_completion_at_ns(self) -> int | None:
        with self._condition:
            return self._last_completion_at_ns

    def release_request(self) -> None:
        """Release all currently blocked calls; repeated calls are harmless."""

        self.release_event.set()

    def reset_events(self) -> None:
        """Reset start/release observability before a new test request.

        A gate cannot be reset while a request is active or after it has been
        closed.  Resetting is explicit so a released event never accidentally
        unblocks a later request.
        """

        with self._condition:
            if self._closed or self._active_count:
                raise ResourceUnavailableError(
                    "advanced fake lifecycle gate cannot reset while active or closed",
                    details={"resource": self.name},
                )
            # Replace rather than clear: asyncio.Event remembers the loop it was
            # first awaited on, and a later unittest/asyncio.run cycle may use a
            # different loop.  Replacement stays under the active-call lock so
            # a worker cannot enter against the old events midway through.
            self.started_event = LifecycleEvent()
            self.release_event = LifecycleEvent()
            self.started = self.started_event
            self.release = self.release_event
            self.request_started_event = self.started_event
            self.request_release_event = self.release_event

    reset = reset_events

    def _begin(
        self,
        request_id: str,
        *,
        block: bool,
        method: str,
    ) -> None:
        _validated_request_id(request_id)
        with self._condition:
            if self._closed or self._close_prepared:
                raise ResourceUnavailableError(
                    "advanced fake resource is closed",
                    details={"resource": self.name},
                )
            self._active_count += 1
        self._call_log.record(
            request_id=request_id,
            component=self.name,
            method=method,
            state=AdvancedRuntimeState.SUCCESS,
        )
        if block:
            self.started_event.set()

    def _finish(self) -> None:
        with self._condition:
            self._active_count -= 1
            self._last_completion_at_ns = monotonic_ns()
            self._condition.notify_all()

    @contextmanager
    def operation(
        self,
        request_id: str,
        *,
        block: bool = False,
    ) -> Iterator[None]:
        self._begin(request_id, block=block, method="operation")
        try:
            if block and not self.release_event.wait_blocking(self.timeout_sec):
                raise BranchTimeoutError(
                    "advanced fake lifecycle gate timed out",
                    details={"resource": self.name},
                )
            yield
        finally:
            self._finish()

    async def execute(self, request_id: str) -> object:
        """Run one lifecycle-only async call until the public release event."""

        self._begin(request_id, block=True, method="execute")
        try:
            try:
                await asyncio.wait_for(
                    self.release_event.wait(),
                    timeout=self.timeout_sec,
                )
            except TimeoutError as exc:
                raise BranchTimeoutError(
                    "advanced fake lifecycle gate timed out",
                    details={"resource": self.name},
                ) from exc
            return self._return_value
        finally:
            self._finish()

    def wait_until_idle_blocking(self, timeout_sec: float | None = None) -> bool:
        if timeout_sec is not None and (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(float(timeout_sec))
            or float(timeout_sec) < 0.0
        ):
            raise ValueError("timeout_sec must be finite and >= 0")
        with self._condition:
            return self._condition.wait_for(
                lambda: self._active_count == 0,
                timeout=None if timeout_sec is None else float(timeout_sec),
            )

    async def wait_until_idle(self, timeout_sec: float | None = None) -> bool:
        return await asyncio.to_thread(self.wait_until_idle_blocking, timeout_sec)

    def _record_close_attempt(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._close_attempt_count += 1
            if self._first_close_attempt_at_ns is None:
                self._first_close_attempt_at_ns = monotonic_ns()
            self.close_called_event.set()

    def _assert_can_close(self, *, record_attempt: bool = True) -> None:
        with self._condition:
            if self._closed:
                return
            if self._close_prepared:
                return
            if record_attempt:
                self._close_attempt_count += 1
                if self._first_close_attempt_at_ns is None:
                    self._first_close_attempt_at_ns = monotonic_ns()
                self.close_called_event.set()
            if self._active_count:
                self._close_attempted_while_active = True
                raise ResourceUnavailableError(
                    "advanced fake resource still has active calls",
                    details={
                        "resource": self.name,
                        "active_count": self._active_count,
                    },
                )
            self._close_prepared = True

    def _cancel_close_preparation(self) -> None:
        with self._condition:
            if not self._closed:
                self._close_prepared = False

    def _commit_close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._active_count:
                self._close_attempted_while_active = True
                raise ResourceUnavailableError(
                    "advanced fake resource still has active calls",
                    details={
                        "resource": self.name,
                        "active_count": self._active_count,
                    },
                )
            self._closed = True
            self._close_prepared = False
            self._close_count = 1
            self._closed_at_ns = monotonic_ns()
            self._condition.notify_all()

    def close(
        self,
        *,
        wait: bool = False,
        timeout_sec: float | None = None,
    ) -> None:
        """Close once all calls finish; successful repeated closes are no-ops."""

        if not isinstance(wait, bool):
            raise TypeError("wait must be a boolean")
        self._record_close_attempt()
        if wait:
            effective_timeout = self.timeout_sec if timeout_sec is None else timeout_sec
            if not self.wait_until_idle_blocking(effective_timeout):
                raise BranchTimeoutError(
                    "advanced fake lifecycle gate did not drain before close",
                    details={"resource": self.name},
                )
        self._assert_can_close(record_attempt=False)
        self._commit_close()


# Discoverable alias used in handoff documentation.
AdvancedLifecycleGate = BlockingLifecycleFake


def _safe_boundary_error(
    exc: DataInfrastructureError,
    *,
    component: str,
    method: str,
) -> DataInfrastructureError:
    details = {"component": component, "method": method}
    retryable = exc.details.get("retryable")
    if isinstance(retryable, bool):
        details["retryable"] = retryable
    if isinstance(exc, BranchTimeoutError):
        return BranchTimeoutError("advanced fake dependency timed out", details=details)
    if isinstance(exc, ResourceUnavailableError):
        return ResourceUnavailableError(
            "advanced fake dependency is unavailable",
            details=details,
        )
    if isinstance(exc, ContractMismatchError):
        return ContractMismatchError(
            "advanced fake dependency violated its contract",
            details=details,
        )
    if isinstance(exc, DimensionMismatchError):
        return DimensionMismatchError(
            "advanced fake dependency returned an incompatible dimension",
            details=details,
        )
    if isinstance(exc, InvalidQueryError):
        return InvalidQueryError(
            "advanced fake dependency rejected invalid input",
            details=details,
        )
    if isinstance(exc, MissingMetadataError):
        return MissingMetadataError(
            "advanced fake dependency could not hydrate metadata",
            details=details,
        )
    return ResourceUnavailableError(
        "advanced fake dependency failed",
        details=details,
    )


class _RuntimeLease:
    """Local open/closed guard for one bundle or request-scoped view."""

    def __init__(self) -> None:
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def ensure_open(self) -> None:
        if self.closed:
            raise ResourceUnavailableError("advanced fake runtime view is closed")

    def close(self) -> None:
        with self._lock:
            self._closed = True


class _RuntimePort:
    def __init__(
        self,
        *,
        component: str,
        state: AdvancedRuntimeState,
        request_id: str,
        call_log: AdvancedRuntimeCallLog,
        lifecycle: BlockingLifecycleFake,
        lease: _RuntimeLease,
    ) -> None:
        self._component = component
        self._state = state
        self._request_id = request_id
        self._call_log = call_log
        self._lifecycle = lifecycle
        self._lease = lease

    @property
    def calls(self) -> tuple[AdvancedRuntimeCall, ...]:
        return tuple(
            call
            for call in self._call_log.snapshot()
            if call.component == self._component
        )

    def _invoke(
        self,
        *,
        method: str,
        operation: Callable[[], _T],
        empty_factory: Callable[[], _T],
        state: AdvancedRuntimeState | None = None,
        request_id: str | None = None,
        identifiers: Sequence[str] = (),
        item_count: int = 0,
        block: bool = False,
    ) -> _T:
        self._lease.ensure_open()
        active_state = state or self._state
        active_request_id = request_id or self._request_id
        self._call_log.record(
            request_id=active_request_id,
            component=self._component,
            method=method,
            state=active_state,
            item_count=item_count,
            identifiers=identifiers,
        )
        with self._lifecycle.operation(active_request_id, block=block):
            if active_state is AdvancedRuntimeState.EMPTY:
                return empty_factory()
            if active_state is AdvancedRuntimeState.TIMEOUT:
                raise BranchTimeoutError(
                    "advanced fake operation timed out",
                    details={"component": self._component, "method": method},
                )
            if active_state is AdvancedRuntimeState.UNAVAILABLE:
                raise ResourceUnavailableError(
                    "advanced fake resource is unavailable",
                    details={"component": self._component, "method": method},
                )
            if active_state is AdvancedRuntimeState.INVALID_REFERENCE:
                raise ContractMismatchError(
                    "advanced fake returned an invalid reference",
                    details={"component": self._component, "method": method},
                )
            try:
                return operation()
            except DataInfrastructureError as exc:
                raise _safe_boundary_error(
                    exc,
                    component=self._component,
                    method=method,
                ) from None
            except Exception:
                raise ResourceUnavailableError(
                    "advanced fake dependency failed unexpectedly",
                    details={"component": self._component, "method": method},
                ) from None


def _validated_string_sequence(
    values: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise InvalidQueryError(f"{field_name} must be a sequence of strings")
    try:
        output = tuple(values)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InvalidQueryError(
            f"{field_name} must be a sequence of strings"
        ) from None
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        for value in output
    ):
        raise InvalidQueryError(
            f"{field_name} must contain normalized non-empty strings"
        )
    return output


def _validated_normalized_string(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise InvalidQueryError(f"{field_name} must be a normalized non-empty string")
    return value


def _validated_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidQueryError(f"{field_name} must be a positive integer")
    return value


def _validated_time_range(start_sec: object, end_sec: object) -> tuple[float, float]:
    values = (start_sec, end_sec)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise InvalidQueryError("ASR time range must contain finite numbers")
    start, end = float(start_sec), float(end_sec)
    if start < 0.0 or end < start:
        raise InvalidQueryError("ASR time range must satisfy 0 <= start_sec <= end_sec")
    return start, end


class AdvancedRuntimeTextEncoder(_RuntimePort):
    def __init__(
        self,
        delegate: TextEncoderPort,
        **kwargs: Any,
    ) -> None:
        super().__init__(component="event_encoder", **kwargs)
        self._delegate = delegate

    @property
    def dimension(self) -> int:
        def read_dimension() -> int:
            dimension = self._delegate.dimension
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 1
            ):
                raise ContractMismatchError(
                    "advanced fake encoder returned an invalid dimension",
                    details={"component": "event_encoder"},
                )
            return dimension

        return self._invoke(
            method="dimension",
            operation=read_dimension,
            empty_factory=lambda: 0,
            state=(
                AdvancedRuntimeState.SUCCESS
                if self._state is AdvancedRuntimeState.EMPTY
                else self._state
            ),
        )

    def encode_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        validated_texts = _validated_string_sequence(texts, field_name="texts")
        if len(set(validated_texts)) != len(validated_texts):
            raise InvalidQueryError("texts must not contain duplicates")

        def collect() -> tuple[tuple[float, ...], ...]:
            try:
                return tuple(
                    tuple(vector)
                    for vector in self._delegate.encode_texts(validated_texts)
                )
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake encoder returned an invalid vector sequence",
                    details={"component": "event_encoder"},
                ) from None

        raw = self._invoke(
            method="encode_texts",
            operation=collect,
            empty_factory=tuple,
            item_count=len(validated_texts),
        )
        try:
            vectors = tuple(tuple(vector) for vector in raw)
        except (TypeError, ValueError):
            raise ContractMismatchError(
                "advanced fake encoder returned an invalid vector sequence",
                details={"component": "event_encoder"},
            ) from None
        if any(
            not vector
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            )
            for vector in vectors
        ):
            raise ContractMismatchError(
                "advanced fake encoder returned invalid vector values",
                details={"component": "event_encoder"},
            )
        expected_count = len(validated_texts)
        if (
            self._state is not AdvancedRuntimeState.EMPTY
            and expected_count
            and len(vectors) != expected_count
        ):
            raise ContractMismatchError(
                "advanced fake encoder returned an unexpected vector count",
                details={"component": "event_encoder"},
            )
        return vectors


class AdvancedRuntimeVisualCorpus(_RuntimePort):
    def __init__(
        self,
        delegate: VisualCorpusPort,
        *,
        block_calls: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(component="visual_corpus", **kwargs)
        self._delegate = delegate
        self._block_calls = block_calls

    def list_video_ids(self) -> tuple[str, ...]:
        def collect() -> tuple[str, ...]:
            try:
                return tuple(self._delegate.list_video_ids())
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake visual corpus returned invalid video IDs",
                    details={"component": "visual_corpus"},
                ) from None

        raw = self._invoke(
            method="list_video_ids",
            operation=collect,
            empty_factory=tuple,
        )
        try:
            values = tuple(raw)
        except TypeError:
            raise ContractMismatchError(
                "advanced fake visual corpus returned invalid video IDs",
                details={"component": "visual_corpus"},
            ) from None
        if any(
            not isinstance(video_id, str)
            or not video_id
            or video_id != video_id.strip()
            for video_id in values
        ):
            raise ContractMismatchError(
                "advanced fake visual corpus returned invalid video IDs",
                details={"component": "visual_corpus"},
            )
        if len(set(values)) != len(values):
            raise ContractMismatchError(
                "advanced fake visual corpus returned duplicate video IDs",
                details={"component": "visual_corpus"},
            )
        return values

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ) -> tuple[tuple[OrderedVisualFrame, ...], ...]:
        validated_video_id = _validated_normalized_string(
            video_id,
            field_name="video_id",
        )
        validated_batch_size = _validated_positive_int(
            batch_size,
            field_name="batch_size",
        )

        def collect() -> tuple[tuple[OrderedVisualFrame, ...], ...]:
            try:
                return tuple(
                    tuple(batch)
                    for batch in self._delegate.iter_ordered_frame_embedding_batches(
                        validated_video_id,
                        validated_batch_size,
                    )
                )
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake visual corpus returned invalid batches",
                    details={"component": "visual_corpus"},
                ) from None

        raw = self._invoke(
            method="iter_ordered_frame_embedding_batches",
            operation=collect,
            empty_factory=tuple,
            identifiers=(validated_video_id,),
            item_count=1,
            block=self._block_calls,
        )
        try:
            batches = tuple(raw)
        except TypeError:
            raise ContractMismatchError(
                "advanced fake visual corpus returned invalid batches",
                details={"component": "visual_corpus"},
            ) from None
        if any(
            any(not isinstance(frame, OrderedVisualFrame) for frame in batch)
            for batch in batches
        ):
            raise ContractMismatchError(
                "advanced fake visual corpus returned invalid frame records",
                details={"component": "visual_corpus"},
            )
        try:
            validate_ordered_visual_stream(validated_video_id, batches)
        except (TypeError, ValueError):
            raise ContractMismatchError(
                "advanced fake visual corpus returned invalid ordering or provenance",
                details={"component": "visual_corpus"},
            ) from None
        return batches


class AdvancedRuntimeMetadataReader(_RuntimePort):
    def __init__(
        self,
        delegate: MetadataReaderPort,
        **kwargs: Any,
    ) -> None:
        super().__init__(component="metadata_reader", **kwargs)
        self._delegate = delegate

    def get_frames_by_ids(
        self,
        frame_ids: Sequence[str],
    ) -> Mapping[str, FrameMetadata]:
        validated_ids = _validated_string_sequence(
            frame_ids,
            field_name="frame_ids",
        )

        def collect() -> Mapping[str, FrameMetadata]:
            try:
                return dict(self._delegate.get_frames_by_ids(validated_ids))
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake metadata reader returned an invalid mapping",
                    details={"component": "metadata_reader"},
                ) from None

        raw = self._invoke(
            method="get_frames_by_ids",
            operation=collect,
            empty_factory=lambda: MappingProxyType({}),
            identifiers=validated_ids,
            item_count=len(validated_ids),
        )
        try:
            output = dict(raw)
        except (TypeError, ValueError):
            raise ContractMismatchError(
                "advanced fake metadata reader returned an invalid mapping",
                details={"component": "metadata_reader"},
            ) from None
        if any(
            not isinstance(frame_id, str)
            or not isinstance(metadata, FrameMetadata)
            or metadata.frame_id != frame_id
            or frame_id not in set(validated_ids)
            for frame_id, metadata in output.items()
        ):
            raise ContractMismatchError(
                "advanced fake metadata reader returned an invalid identity",
                details={"component": "metadata_reader"},
            )
        return MappingProxyType(output)

    def get_videos_by_ids(
        self,
        video_ids: Sequence[str],
    ) -> Mapping[str, VideoMetadata]:
        validated_ids = _validated_string_sequence(video_ids, field_name="video_ids")

        def collect() -> Mapping[str, VideoMetadata]:
            try:
                return dict(self._delegate.get_videos_by_ids(validated_ids))
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake metadata reader returned an invalid video mapping",
                    details={"component": "metadata_reader"},
                ) from None

        raw = self._invoke(
            method="get_videos_by_ids",
            operation=collect,
            empty_factory=lambda: MappingProxyType({}),
            identifiers=validated_ids,
            item_count=len(validated_ids),
        )
        try:
            output = dict(raw)
        except (TypeError, ValueError):
            raise ContractMismatchError(
                "advanced fake metadata reader returned an invalid video mapping",
                details={"component": "metadata_reader"},
            ) from None
        if any(
            not isinstance(video_id, str)
            or not isinstance(metadata, VideoMetadata)
            or metadata.video_id != video_id
            or video_id not in set(validated_ids)
            for video_id, metadata in output.items()
        ):
            raise ContractMismatchError(
                "advanced fake metadata reader returned an invalid video identity",
                details={"component": "metadata_reader"},
            )
        return MappingProxyType(output)

    def get_ordered_frames_by_video(
        self,
        video_id: str,
    ) -> tuple[FrameMetadata, ...]:
        validated_video_id = _validated_normalized_string(
            video_id,
            field_name="video_id",
        )
        def collect() -> tuple[FrameMetadata, ...]:
            try:
                return tuple(
                    self._delegate.get_ordered_frames_by_video(validated_video_id)
                )
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake metadata reader returned an invalid sequence",
                    details={"component": "metadata_reader"},
                ) from None

        raw = self._invoke(
            method="get_ordered_frames_by_video",
            operation=collect,
            empty_factory=tuple,
            identifiers=(validated_video_id,),
            item_count=1,
        )
        try:
            output = tuple(raw)
        except TypeError:
            raise ContractMismatchError(
                "advanced fake metadata reader returned an invalid sequence",
                details={"component": "metadata_reader"},
            ) from None
        if any(not isinstance(metadata, FrameMetadata) for metadata in output):
            raise ContractMismatchError(
                "advanced fake metadata reader returned an invalid record",
                details={"component": "metadata_reader"},
            )
        if (
            any(metadata.video_id != validated_video_id for metadata in output)
            or len({metadata.frame_id for metadata in output}) != len(output)
            or tuple(output)
            != tuple(
                sorted(
                    output,
                    key=lambda metadata: (
                        metadata.timestamp_sec,
                        metadata.frame_id,
                    ),
                )
            )
        ):
            raise ContractMismatchError(
                "advanced fake metadata reader returned invalid ordering or provenance",
                details={"component": "metadata_reader"},
            )
        return output


class AdvancedRuntimeEvidenceHydrator(_RuntimePort):
    def __init__(
        self,
        delegate: EvidenceHydrationPort,
        *,
        ocr_state: AdvancedRuntimeState,
        asr_state: AdvancedRuntimeState,
        summary_state: AdvancedRuntimeState,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            component="evidence_hydrator",
            state=AdvancedRuntimeState.SUCCESS,
            **kwargs,
        )
        self._delegate = delegate
        self._ocr_state = ocr_state
        self._asr_state = asr_state
        self._summary_state = summary_state

    def get_ocr_evidence(
        self,
        frame_ids: Sequence[str],
    ) -> tuple[OCREvidence, ...]:
        validated_ids = _validated_string_sequence(
            frame_ids,
            field_name="frame_ids",
        )
        def collect() -> tuple[OCREvidence, ...]:
            try:
                return tuple(self._delegate.get_ocr_evidence(validated_ids))
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake dependency returned an invalid sequence",
                    details={"component": "evidence_hydrator"},
                ) from None

        raw = self._invoke(
            method="get_ocr_evidence",
            operation=collect,
            empty_factory=tuple,
            state=self._ocr_state,
            identifiers=validated_ids,
            item_count=len(validated_ids),
        )
        output = _typed_tuple(raw, OCREvidence, component="evidence_hydrator")
        if (
            any(record.frame_id not in set(validated_ids) for record in output)
            or len({record.evidence_id for record in output}) != len(output)
        ):
            raise ContractMismatchError(
                "advanced fake OCR evidence has invalid provenance",
                details={"component": "evidence_hydrator"},
            )
        return output

    def get_asr_evidence(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
    ) -> tuple[ASREvidence, ...]:
        validated_video_id = _validated_normalized_string(
            video_id,
            field_name="video_id",
        )
        start, end = _validated_time_range(start_sec, end_sec)
        def collect() -> tuple[ASREvidence, ...]:
            try:
                return tuple(
                    self._delegate.get_asr_evidence(
                        validated_video_id,
                        start,
                        end,
                    )
                )
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake dependency returned an invalid sequence",
                    details={"component": "evidence_hydrator"},
                ) from None

        raw = self._invoke(
            method="get_asr_evidence",
            operation=collect,
            empty_factory=tuple,
            state=self._asr_state,
            identifiers=(validated_video_id,),
            item_count=1,
        )
        output = _typed_tuple(raw, ASREvidence, component="evidence_hydrator")
        if (
            any(
                record.video_id != validated_video_id
                or record.end_time_sec < start
                or record.start_time_sec > end
                for record in output
            )
            or len({record.evidence_id for record in output}) != len(output)
        ):
            raise ContractMismatchError(
                "advanced fake ASR evidence has invalid provenance",
                details={"component": "evidence_hydrator"},
            )
        return output

    def get_summary_evidence(
        self,
        video_ids: Sequence[str],
    ) -> tuple[SummaryEvidence, ...]:
        validated_ids = _validated_string_sequence(
            video_ids,
            field_name="video_ids",
        )
        def collect() -> tuple[SummaryEvidence, ...]:
            try:
                return tuple(self._delegate.get_summary_evidence(validated_ids))
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake dependency returned an invalid sequence",
                    details={"component": "evidence_hydrator"},
                ) from None

        raw = self._invoke(
            method="get_summary_evidence",
            operation=collect,
            empty_factory=tuple,
            state=self._summary_state,
            identifiers=validated_ids,
            item_count=len(validated_ids),
        )
        output = _typed_tuple(raw, SummaryEvidence, component="evidence_hydrator")
        if (
            any(record.video_id not in set(validated_ids) for record in output)
            or len({record.evidence_id for record in output}) != len(output)
        ):
            raise ContractMismatchError(
                "advanced fake summary evidence has invalid provenance",
                details={"component": "evidence_hydrator"},
            )
        return output


def _typed_tuple(
    values: object,
    expected_type: type[_T],
    *,
    component: str,
) -> tuple[_T, ...]:
    try:
        output = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        raise ContractMismatchError(
            "advanced fake dependency returned an invalid sequence",
            details={"component": component},
        ) from None
    if any(not isinstance(value, expected_type) for value in output):
        raise ContractMismatchError(
            "advanced fake dependency returned an invalid record type",
            details={"component": component},
        )
    return output


class AdvancedRuntimeImageResolver(_RuntimePort):
    def __init__(
        self,
        delegate: ImageResolverPort,
        **kwargs: Any,
    ) -> None:
        super().__init__(component="image_resolver", **kwargs)
        self._delegate = delegate

    def resolve_images(
        self,
        frame_ids: Sequence[str],
    ) -> Mapping[str, ImageEvidence]:
        validated_ids = _validated_string_sequence(
            frame_ids,
            field_name="frame_ids",
        )

        def collect() -> Mapping[str, ImageEvidence]:
            try:
                return dict(self._delegate.resolve_images(validated_ids))
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError):
                raise ContractMismatchError(
                    "advanced fake image resolver returned an invalid mapping",
                    details={"component": "image_resolver"},
                ) from None

        raw = self._invoke(
            method="resolve_images",
            operation=collect,
            empty_factory=lambda: MappingProxyType({}),
            identifiers=validated_ids,
            item_count=len(validated_ids),
        )
        try:
            output = dict(raw)
        except (TypeError, ValueError):
            raise ContractMismatchError(
                "advanced fake image resolver returned an invalid mapping",
                details={"component": "image_resolver"},
            ) from None
        if any(
            not isinstance(frame_id, str)
            or not isinstance(evidence, ImageEvidence)
            or evidence.frame_id != frame_id
            or frame_id not in set(validated_ids)
            for frame_id, evidence in output.items()
        ):
            raise ContractMismatchError(
                "advanced fake image resolver returned an invalid reference",
                details={"component": "image_resolver"},
            )
        return MappingProxyType(output)


class AdvancedRuntimeVLM(_RuntimePort):
    def __init__(
        self,
        delegate: VLMPort,
        *,
        block_calls: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(component="vlm", **kwargs)
        self._delegate = delegate
        self._block_calls = block_calls

    def answer(self, request: VLMRequest) -> Any:
        if not isinstance(request, VLMRequest):
            raise InvalidQueryError("request must be a validated VLMRequest")
        evidence_ids = tuple(item.evidence_id for item in request.evidence)

        def insufficient() -> VLMResponse:
            return VLMResponse(
                status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                answer_type=request.question.answer_type,
                confidence=VLMConfidence.LOW,
                evidence_ids=(),
            )

        def collect() -> object:
            try:
                return _immutable_test_value(self._delegate.answer(request))
            except DataInfrastructureError:
                raise
            except (TypeError, ValueError, RecursionError):
                raise ContractMismatchError(
                    "advanced fake VLM returned an invalid value",
                    details={"component": "vlm"},
                ) from None

        # Preserve malformed values for C's negative-path contract tests, but
        # snapshot mutable containers before the lifecycle lease is released.
        return self._invoke(
            method="answer",
            operation=collect,
            empty_factory=insufficient,
            request_id=request.request_id,
            identifiers=evidence_ids,
            item_count=len(evidence_ids),
            block=self._block_calls,
        )


class AdvancedRuntimeBundle:
    """Public, fake-only handoff consumed by Wave 3 composition tests."""

    def __init__(
        self,
        *,
        config: AdvancedRuntimeConfig,
        fixture: AdvancedModesFixture,
        text_encoder: AdvancedRuntimeTextEncoder,
        visual_corpus: AdvancedRuntimeVisualCorpus,
        metadata_reader: AdvancedRuntimeMetadataReader,
        evidence_hydrator: AdvancedRuntimeEvidenceHydrator,
        image_resolver: AdvancedRuntimeImageResolver,
        vlm: AdvancedRuntimeVLM,
        call_log: AdvancedRuntimeCallLog,
        trake_lifecycle: BlockingLifecycleFake,
        vqa_lifecycle: BlockingLifecycleFake,
        delegates: Mapping[str, object],
        lease: _RuntimeLease,
        owns_lifecycle: bool,
    ) -> None:
        self._config = config
        self._fixture = fixture
        self._text_encoder = text_encoder
        self._visual_corpus = visual_corpus
        self._metadata_reader = metadata_reader
        self._evidence_hydrator = evidence_hydrator
        self._image_resolver = image_resolver
        self._vlm = vlm
        self._call_log = call_log
        self._trake_lifecycle = trake_lifecycle
        self._vqa_lifecycle = vqa_lifecycle
        self._delegates = MappingProxyType(dict(delegates))
        self._lease = lease
        self._owns_lifecycle = owns_lifecycle
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._connected = True
        self._closed = False
        self._close_count = 0

    @property
    def config(self) -> AdvancedRuntimeConfig:
        return self._config

    @property
    def request_id(self) -> str:
        return self._config.request_id

    @property
    def fixture(self) -> AdvancedModesFixture:
        return self._fixture

    @property
    def text_encoder(self) -> AdvancedRuntimeTextEncoder:
        return self._text_encoder

    @property
    def event_encoder_port(self) -> AdvancedRuntimeTextEncoder:
        return self._text_encoder

    @property
    def event_encoder(self) -> AdvancedRuntimeTextEncoder:
        return self._text_encoder

    @property
    def visual_corpus(self) -> AdvancedRuntimeVisualCorpus:
        return self._visual_corpus

    @property
    def visual_corpus_port(self) -> AdvancedRuntimeVisualCorpus:
        return self._visual_corpus

    @property
    def corpus(self) -> AdvancedRuntimeVisualCorpus:
        return self._visual_corpus

    @property
    def metadata_reader(self) -> AdvancedRuntimeMetadataReader:
        return self._metadata_reader

    @property
    def metadata_reader_port(self) -> AdvancedRuntimeMetadataReader:
        return self._metadata_reader

    @property
    def metadata(self) -> AdvancedRuntimeMetadataReader:
        return self._metadata_reader

    @property
    def evidence_hydrator(self) -> AdvancedRuntimeEvidenceHydrator:
        return self._evidence_hydrator

    @property
    def evidence_hydration_port(self) -> AdvancedRuntimeEvidenceHydrator:
        return self._evidence_hydrator

    @property
    def evidence_reader(self) -> AdvancedRuntimeEvidenceHydrator:
        return self._evidence_hydrator

    @property
    def image_resolver(self) -> AdvancedRuntimeImageResolver:
        return self._image_resolver

    @property
    def image_resolver_port(self) -> AdvancedRuntimeImageResolver:
        return self._image_resolver

    @property
    def image(self) -> AdvancedRuntimeImageResolver:
        return self._image_resolver

    @property
    def vlm(self) -> AdvancedRuntimeVLM:
        return self._vlm

    @property
    def vlm_port(self) -> AdvancedRuntimeVLM:
        return self._vlm

    @property
    def call_log(self) -> AdvancedRuntimeCallLog:
        return self._call_log

    @property
    def calls(self) -> tuple[AdvancedRuntimeCall, ...]:
        return self._call_log.snapshot()

    @property
    def call_log_snapshot(self) -> tuple[AdvancedRuntimeCall, ...]:
        return self._call_log.snapshot()

    @property
    def trake_lifecycle(self) -> BlockingLifecycleFake:
        return self._trake_lifecycle

    @property
    def trake_gate(self) -> BlockingLifecycleFake:
        return self._trake_lifecycle

    @property
    def vqa_lifecycle(self) -> BlockingLifecycleFake:
        return self._vqa_lifecycle

    @property
    def vqa_gate(self) -> BlockingLifecycleFake:
        return self._vqa_lifecycle

    @property
    def lifecycle(self) -> Mapping[str, BlockingLifecycleFake]:
        return MappingProxyType(
            {"trake": self._trake_lifecycle, "vqa": self._vqa_lifecycle}
        )

    @property
    def resources(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "event_encoder": self._text_encoder,
                "visual_corpus": self._visual_corpus,
                "metadata_reader": self._metadata_reader,
                "evidence_hydrator": self._evidence_hydrator,
                "image_resolver": self._image_resolver,
                "vlm": self._vlm,
            }
        )

    @property
    def trake_started_event(self) -> LifecycleEvent:
        return self._trake_lifecycle.started_event

    @property
    def trake_release_event(self) -> LifecycleEvent:
        return self._trake_lifecycle.release_event

    @property
    def vqa_started_event(self) -> LifecycleEvent:
        return self._vqa_lifecycle.started_event

    @property
    def vqa_release_event(self) -> LifecycleEvent:
        return self._vqa_lifecycle.release_event

    @property
    def closed(self) -> bool:
        with self._lock:
            locally_closed = self._closed
        return (
            locally_closed
            or self._trake_lifecycle.closed
            or self._vqa_lifecycle.closed
        )

    @property
    def is_closed(self) -> bool:
        return self.closed

    @property
    def close_count(self) -> int:
        with self._lock:
            return self._close_count

    def release_trake(self) -> None:
        self._trake_lifecycle.release_request()

    def release_vqa(self) -> None:
        self._vqa_lifecycle.release_request()

    def release_all(self) -> None:
        self.release_trake()
        self.release_vqa()

    async def wait_until_idle(self, timeout_sec: float | None = None) -> bool:
        """Await both resource groups becoming idle."""

        started = monotonic_ns()
        if not await self._trake_lifecycle.wait_until_idle(timeout_sec):
            return False
        if timeout_sec is None:
            return await self._vqa_lifecycle.wait_until_idle(None)
        elapsed = (monotonic_ns() - started) / 1_000_000_000
        remaining = max(0.0, float(timeout_sec) - elapsed)
        return await self._vqa_lifecycle.wait_until_idle(remaining)

    wait_for_drain = wait_until_idle

    def for_request(self, request_id: str) -> "AdvancedRuntimeBundle":
        """Create an isolated request-scoped view with the same immutable data.

        Frozen port signatures do not carry a request ID.  A child bundle is the
        explicit correlation boundary for concurrent composition tests; it
        avoids mutable global/context state, gives each request its own log and
        local lease, and shares the root lifecycle gates so shutdown sees every
        active request.
        """

        if self.closed:
            raise ResourceUnavailableError("advanced fake runtime is closed")
        return build_advanced_runtime_bundle(
            config=self._config.for_request(request_id),
            fixture=self._fixture,
            text_encoder=self._delegates["text_encoder"],  # type: ignore[arg-type]
            visual_corpus=self._delegates["visual_corpus"],  # type: ignore[arg-type]
            metadata_reader=self._delegates["metadata_reader"],  # type: ignore[arg-type]
            evidence_hydrator=self._delegates["evidence_hydrator"],  # type: ignore[arg-type]
            image_resolver=self._delegates["image_resolver"],  # type: ignore[arg-type]
            vlm=self._delegates["vlm"],  # type: ignore[arg-type]
            _trake_lifecycle=self._trake_lifecycle,
            _vqa_lifecycle=self._vqa_lifecycle,
            _owns_lifecycle=False,
        )

    request_scope = for_request

    def connect(self) -> None:
        """No-I/O ManagedResource-compatible connect hook."""

        if self._trake_lifecycle.closed or self._vqa_lifecycle.closed:
            raise ResourceUnavailableError("advanced fake runtime is closed")
        with self._lock:
            if self._closed:
                raise ResourceUnavailableError("advanced fake runtime is closed")
            self._connected = True

    def health_check(self) -> None:
        """Fail deterministically for configured non-ready resource states."""

        if self._trake_lifecycle.closed or self._vqa_lifecycle.closed:
            raise ResourceUnavailableError("advanced fake runtime is not connected")
        with self._lock:
            if self._closed or not self._connected:
                raise ResourceUnavailableError("advanced fake runtime is not connected")
        component_states = (
            ("event_encoder", self._config.encoder_state),
            ("visual_corpus", self._config.visual_state),
            ("metadata_reader", self._config.metadata_state),
            ("ocr_evidence", self._config.ocr_state),
            ("asr_evidence", self._config.asr_state),
            ("summary_evidence", self._config.summary_state),
            ("image_resolver", self._config.image_state),
            ("vlm", self._config.vlm_state),
        )
        for component, state in component_states:
            if state is AdvancedRuntimeState.TIMEOUT:
                raise BranchTimeoutError(
                    "advanced fake readiness probe timed out",
                    details={"component": component},
                )
            if state is AdvancedRuntimeState.UNAVAILABLE:
                raise ResourceUnavailableError(
                    "advanced fake resource is unavailable",
                    details={"component": component},
                )
            if state is AdvancedRuntimeState.INVALID_REFERENCE:
                raise ContractMismatchError(
                    "advanced fake resource has an invalid reference",
                    details={"component": component},
                )

    def close(
        self,
        *,
        wait: bool = False,
        timeout_sec: float | None = None,
    ) -> None:
        """Close both fake resource groups after active requests have drained."""

        if not isinstance(wait, bool):
            raise TypeError("wait must be a boolean")
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
            if not self._owns_lifecycle:
                self._lease.close()
                with self._lock:
                    self._connected = False
                    self._closed = True
                    self._close_count = 1
                return
            if wait:
                effective_timeout = (
                    self._trake_lifecycle.timeout_sec
                    if timeout_sec is None
                    else timeout_sec
                )
                wait_started = monotonic_ns()
                if not self._trake_lifecycle.wait_until_idle_blocking(
                    effective_timeout
                ):
                    raise BranchTimeoutError(
                        "advanced TRAKE fake did not drain before close",
                        details={"resource": "advanced_trake"},
                    )
                elapsed = (monotonic_ns() - wait_started) / 1_000_000_000
                remaining = max(0.0, float(effective_timeout) - elapsed)
                if not self._vqa_lifecycle.wait_until_idle_blocking(
                    remaining
                ):
                    raise BranchTimeoutError(
                        "advanced VQA fake did not drain before close",
                        details={"resource": "advanced_vqa"},
                    )
            # Two-phase preflight avoids partially closing one resource group.
            prepared: list[BlockingLifecycleFake] = []
            try:
                for lifecycle in (self._vqa_lifecycle, self._trake_lifecycle):
                    lifecycle._assert_can_close()
                    prepared.append(lifecycle)
            except Exception:
                for lifecycle in prepared:
                    lifecycle._cancel_close_preparation()
                raise
            self._vqa_lifecycle._commit_close()
            self._trake_lifecycle._commit_close()
            self._lease.close()
            with self._lock:
                if self._closed:
                    return
                self._connected = False
                self._closed = True
                self._close_count = 1

    def __enter__(self) -> "AdvancedRuntimeBundle":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    async def __aenter__(self) -> "AdvancedRuntimeBundle":
        self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()


def _require_protocol(value: object, protocol: type[Any], *, name: str) -> None:
    if not isinstance(value, protocol):
        raise TypeError(f"{name} must implement its frozen runtime protocol")


def _configured(
    config: AdvancedRuntimeConfig | None,
    *,
    state: AdvancedRuntimeState | str | None,
    request_id: str | None,
    encoder_state: AdvancedRuntimeState | str | None,
    visual_state: AdvancedRuntimeState | str | None,
    metadata_state: AdvancedRuntimeState | str | None,
    evidence_state: AdvancedRuntimeState | str | None,
    ocr_state: AdvancedRuntimeState | str | None,
    asr_state: AdvancedRuntimeState | str | None,
    summary_state: AdvancedRuntimeState | str | None,
    image_state: AdvancedRuntimeState | str | None,
    vlm_state: AdvancedRuntimeState | str | None,
    block_trake: bool | None,
    block_vqa: bool | None,
    block_timeout_sec: float | None,
) -> AdvancedRuntimeConfig:
    active = AdvancedRuntimeConfig() if config is None else config
    if not isinstance(active, AdvancedRuntimeConfig):
        raise TypeError("config must be an AdvancedRuntimeConfig or None")
    values: dict[str, object] = {}
    if state is not None:
        shared_state = _coerce_state(state, field_name="state")
        values.update(
            encoder_state=shared_state,
            visual_state=shared_state,
            metadata_state=shared_state,
            ocr_state=shared_state,
            asr_state=shared_state,
            summary_state=shared_state,
            image_state=shared_state,
            vlm_state=shared_state,
        )
    if request_id is not None:
        values["request_id"] = request_id
    for field_name, value in (
        ("encoder_state", encoder_state),
        ("visual_state", visual_state),
        ("metadata_state", metadata_state),
        ("image_state", image_state),
        ("vlm_state", vlm_state),
    ):
        if value is not None:
            values[field_name] = _coerce_state(value, field_name=field_name)
    if evidence_state is not None:
        shared_evidence_state = _coerce_state(
            evidence_state,
            field_name="evidence_state",
        )
        values.update(
            ocr_state=shared_evidence_state,
            asr_state=shared_evidence_state,
            summary_state=shared_evidence_state,
        )
    for field_name, value in (
        ("ocr_state", ocr_state),
        ("asr_state", asr_state),
        ("summary_state", summary_state),
    ):
        if value is not None:
            values[field_name] = _coerce_state(value, field_name=field_name)
    if block_trake is not None:
        values["block_trake"] = block_trake
    if block_vqa is not None:
        values["block_vqa"] = block_vqa
    if block_timeout_sec is not None:
        values["block_timeout_sec"] = block_timeout_sec
    return replace(active, **values)


def build_advanced_runtime_bundle(
    config: AdvancedRuntimeConfig | None = None,
    *,
    fixture: AdvancedModesFixture | None = None,
    text_encoder: TextEncoderPort | None = None,
    visual_corpus: VisualCorpusPort | None = None,
    metadata_reader: MetadataReaderPort | None = None,
    evidence_hydrator: EvidenceHydrationPort | None = None,
    image_resolver: ImageResolverPort | None = None,
    vlm: VLMPort | None = None,
    state: AdvancedRuntimeState | str | None = None,
    request_id: str | None = None,
    encoder_state: AdvancedRuntimeState | str | None = None,
    event_encoder_state: AdvancedRuntimeState | str | None = None,
    visual_state: AdvancedRuntimeState | str | None = None,
    visual_corpus_state: AdvancedRuntimeState | str | None = None,
    metadata_state: AdvancedRuntimeState | str | None = None,
    metadata_reader_state: AdvancedRuntimeState | str | None = None,
    evidence_state: AdvancedRuntimeState | str | None = None,
    evidence_hydration_state: AdvancedRuntimeState | str | None = None,
    ocr_state: AdvancedRuntimeState | str | None = None,
    asr_state: AdvancedRuntimeState | str | None = None,
    summary_state: AdvancedRuntimeState | str | None = None,
    image_state: AdvancedRuntimeState | str | None = None,
    image_resolver_state: AdvancedRuntimeState | str | None = None,
    vlm_state: AdvancedRuntimeState | str | None = None,
    vlm_behavior_state: AdvancedRuntimeState | str | None = None,
    block_trake: bool | None = None,
    block_vqa: bool | None = None,
    block_timeout_sec: float | None = None,
    _trake_lifecycle: BlockingLifecycleFake | None = None,
    _vqa_lifecycle: BlockingLifecycleFake | None = None,
    _owns_lifecycle: bool = True,
) -> AdvancedRuntimeBundle:
    """Build a fresh fake-only bundle.

    Explicit dependency parameters let B/C inject coherent fake-index data
    whose canonical IDs already match this bundle's evidence.  The factory
    never rewrites IDs or reaches into a database, provider, network, or image
    path.
    """

    active_config = _configured(
        config,
        state=state,
        request_id=request_id,
        encoder_state=(
            encoder_state if encoder_state is not None else event_encoder_state
        ),
        visual_state=(
            visual_state if visual_state is not None else visual_corpus_state
        ),
        metadata_state=(
            metadata_state
            if metadata_state is not None
            else metadata_reader_state
        ),
        evidence_state=(
            evidence_state
            if evidence_state is not None
            else evidence_hydration_state
        ),
        ocr_state=ocr_state,
        asr_state=asr_state,
        summary_state=summary_state,
        image_state=(
            image_state if image_state is not None else image_resolver_state
        ),
        vlm_state=(
            vlm_state if vlm_state is not None else vlm_behavior_state
        ),
        block_trake=block_trake,
        block_vqa=block_vqa,
        block_timeout_sec=block_timeout_sec,
    )
    active_fixture = fixture if fixture is not None else build_advanced_modes_fixture()
    if not isinstance(active_fixture, AdvancedModesFixture):
        raise TypeError("fixture must be an AdvancedModesFixture or None")

    encoder_delegate = (
        text_encoder if text_encoder is not None else active_fixture.text_encoder()
    )
    corpus_delegate = (
        visual_corpus if visual_corpus is not None else active_fixture.visual_corpus()
    )
    metadata_delegate = (
        metadata_reader if metadata_reader is not None else active_fixture.metadata()
    )
    evidence_delegate = (
        evidence_hydrator
        if evidence_hydrator is not None
        else active_fixture.evidence_hydrator()
    )
    image_delegate = (
        image_resolver
        if image_resolver is not None
        else active_fixture.image_resolver()
    )
    vlm_delegate = vlm if vlm is not None else active_fixture.vlm()

    _require_protocol(encoder_delegate, TextEncoderPort, name="text_encoder")
    _require_protocol(corpus_delegate, VisualCorpusPort, name="visual_corpus")
    _require_protocol(metadata_delegate, MetadataReaderPort, name="metadata_reader")
    _require_protocol(
        evidence_delegate,
        EvidenceHydrationPort,
        name="evidence_hydrator",
    )
    _require_protocol(image_delegate, ImageResolverPort, name="image_resolver")
    _require_protocol(vlm_delegate, VLMPort, name="vlm")

    if not isinstance(_owns_lifecycle, bool):
        raise TypeError("_owns_lifecycle must be a boolean")
    call_log = AdvancedRuntimeCallLog()
    lease = _RuntimeLease()
    trake_lifecycle = _trake_lifecycle or BlockingLifecycleFake(
        "advanced_trake",
        timeout_sec=active_config.block_timeout_sec,
    )
    vqa_lifecycle = _vqa_lifecycle or BlockingLifecycleFake(
        "advanced_vqa",
        timeout_sec=active_config.block_timeout_sec,
    )
    if not isinstance(trake_lifecycle, BlockingLifecycleFake):
        raise TypeError("_trake_lifecycle must be a BlockingLifecycleFake")
    if not isinstance(vqa_lifecycle, BlockingLifecycleFake):
        raise TypeError("_vqa_lifecycle must be a BlockingLifecycleFake")
    common = {
        "request_id": active_config.request_id,
        "call_log": call_log,
        "lease": lease,
    }
    runtime_encoder = AdvancedRuntimeTextEncoder(
        encoder_delegate,
        state=active_config.encoder_state,
        lifecycle=trake_lifecycle,
        **common,
    )
    runtime_corpus = AdvancedRuntimeVisualCorpus(
        corpus_delegate,
        state=active_config.visual_state,
        lifecycle=trake_lifecycle,
        block_calls=active_config.block_trake,
        **common,
    )
    runtime_metadata = AdvancedRuntimeMetadataReader(
        metadata_delegate,
        state=active_config.metadata_state,
        lifecycle=vqa_lifecycle,
        **common,
    )
    runtime_evidence = AdvancedRuntimeEvidenceHydrator(
        evidence_delegate,
        ocr_state=active_config.ocr_state,
        asr_state=active_config.asr_state,
        summary_state=active_config.summary_state,
        lifecycle=vqa_lifecycle,
        **common,
    )
    runtime_images = AdvancedRuntimeImageResolver(
        image_delegate,
        state=active_config.image_state,
        lifecycle=vqa_lifecycle,
        **common,
    )
    runtime_vlm = AdvancedRuntimeVLM(
        vlm_delegate,
        state=active_config.vlm_state,
        lifecycle=vqa_lifecycle,
        block_calls=active_config.block_vqa,
        **common,
    )
    return AdvancedRuntimeBundle(
        config=active_config,
        fixture=active_fixture,
        text_encoder=runtime_encoder,
        visual_corpus=runtime_corpus,
        metadata_reader=runtime_metadata,
        evidence_hydrator=runtime_evidence,
        image_resolver=runtime_images,
        vlm=runtime_vlm,
        call_log=call_log,
        trake_lifecycle=trake_lifecycle,
        vqa_lifecycle=vqa_lifecycle,
        delegates={
            "text_encoder": encoder_delegate,
            "visual_corpus": corpus_delegate,
            "metadata_reader": metadata_delegate,
            "evidence_hydrator": evidence_delegate,
            "image_resolver": image_delegate,
            "vlm": vlm_delegate,
        },
        lease=lease,
        owns_lifecycle=_owns_lifecycle,
    )


def build_happy_path_advanced_runtime_bundle(
    *,
    request_id: str = "advanced-wave3-happy",
) -> AdvancedRuntimeBundle:
    """Named happy-path factory for C's composition handoff."""

    return build_advanced_runtime_bundle(request_id=request_id)


create_advanced_runtime_bundle = build_advanced_runtime_bundle


__all__ = [
    "AdvancedLifecycleGate",
    "AdvancedRuntimeBundle",
    "AdvancedRuntimeCall",
    "AdvancedRuntimeCallLog",
    "AdvancedRuntimeConfig",
    "AdvancedRuntimeEvidenceHydrator",
    "AdvancedRuntimeImageResolver",
    "AdvancedRuntimeMetadataReader",
    "AdvancedRuntimeState",
    "AdvancedRuntimeTextEncoder",
    "AdvancedRuntimeVLM",
    "AdvancedRuntimeVisualCorpus",
    "BlockingLifecycleFake",
    "FakeRuntimeState",
    "LifecycleEvent",
    "RuntimeCall",
    "build_advanced_runtime_bundle",
    "build_happy_path_advanced_runtime_bundle",
    "create_advanced_runtime_bundle",
]
