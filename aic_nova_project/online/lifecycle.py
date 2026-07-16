"""Connection lifecycle and health aggregation without retry-policy assumptions."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import Field

from .domain.base import NonEmptyStr, StrictFrozenModel


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ManagedResource(Protocol):
    def connect(self) -> None: ...

    def close(self) -> None: ...

    def health_check(self) -> None: ...


class ComponentHealth(StrictFrozenModel):
    name: NonEmptyStr
    required: bool
    healthy: bool
    message: str = ""


class InfrastructureHealth(StrictFrozenModel):
    status: HealthStatus
    components: tuple[ComponentHealth, ...] = Field(default=())


class InfrastructureLifecycle:
    """Creates one long-lived connection per registered adapter.

    Retry, pooling and circuit-breaker behavior intentionally remain outside this
    class until OQ-021 is approved.
    """

    def __init__(self) -> None:
        self._resources: list[tuple[str, ManagedResource, bool]] = []
        self._started: list[ManagedResource] = []

    def register(self, name: str, resource: ManagedResource, *, required: bool) -> None:
        if not name.strip():
            raise ValueError("resource name must not be empty")
        if any(existing == name for existing, _, _ in self._resources):
            raise ValueError(f"duplicate resource name: {name}")
        self._resources.append((name, resource, required))

    def start(self) -> InfrastructureHealth:
        for _, resource, _ in self._resources:
            if resource in self._started:
                continue
            try:
                resource.connect()
            except Exception:
                continue
            self._started.append(resource)
        return self.health()

    def health(self) -> InfrastructureHealth:
        components: list[ComponentHealth] = []
        for name, resource, required in self._resources:
            try:
                resource.health_check()
            except Exception as exc:
                components.append(
                    ComponentHealth(
                        name=name,
                        required=required,
                        healthy=False,
                        message=str(exc),
                    )
                )
            else:
                components.append(ComponentHealth(name=name, required=required, healthy=True))

        if any(not item.healthy and item.required for item in components):
            status = HealthStatus.UNHEALTHY
        elif any(not item.healthy for item in components):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        return InfrastructureHealth(status=status, components=tuple(components))

    def close(self) -> None:
        first_error: Exception | None = None
        for resource in reversed(self._started):
            try:
                resource.close()
            except Exception as exc:
                first_error = first_error or exc
        self._started.clear()
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "InfrastructureLifecycle":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
