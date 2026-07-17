"""Small lifecycle guard for concurrent read adapters."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator

from online.domain.errors import ResourceUnavailableError


class ConcurrentReadGuard:
    """Allow parallel reads while preventing close/read overlap.

    ``close()`` is intentionally rejected while a read is active. Serving code
    must stop accepting work, drain its executor, and only then close adapters.
    """

    def __init__(self, resource: str) -> None:
        self._resource = resource
        self._condition = threading.Condition(threading.RLock())
        self._active_reads = 0
        self._closing = False

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._condition:
            if self._closing:
                raise ResourceUnavailableError(
                    "Adapter is closing and cannot accept new reads",
                    details={"resource": self._resource},
                )
            self._active_reads += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_reads -= 1
                self._condition.notify_all()

    def begin_close(self) -> None:
        with self._condition:
            if self._active_reads:
                raise ResourceUnavailableError(
                    "Cannot close adapter while read calls are active",
                    details={
                        "resource": self._resource,
                        "active_reads": self._active_reads,
                    },
                )
            self._closing = True

    def end_close(self) -> None:
        with self._condition:
            self._closing = False
