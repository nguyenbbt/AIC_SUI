"""Thin TRAKE mode boundary over the Wave-2 service."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from online.domain.errors import ContractMismatchError
from online.domain.trake import TRAKEDiagnostics, TRAKEQuery, TRAKEVideoResult
from online.trake.service import TRAKEExecution


@runtime_checkable
class TRAKEServicePort(Protocol):
    async def execute(self, query: TRAKEQuery) -> TRAKEExecution: ...


class TRAKEModeAdapter:
    """Validate the mode boundary and delegate DANTE exactly once."""

    def __init__(self, service: TRAKEServicePort) -> None:
        if not isinstance(service, TRAKEServicePort):
            raise TypeError("service must implement TRAKEServicePort")
        self._service = service

    async def execute(self, query: TRAKEQuery) -> TRAKEExecution:
        if not isinstance(query, TRAKEQuery):
            raise ContractMismatchError("query must be a validated TRAKEQuery")
        execution = await self._service.execute(query)
        if not isinstance(execution, TRAKEExecution):
            raise ContractMismatchError("TRAKE service returned an invalid execution")
        if execution.query_id != query.query_id:
            raise ContractMismatchError("TRAKE service changed query_id")
        return execution

    async def search(self, query: TRAKEQuery) -> tuple[TRAKEVideoResult, ...]:
        return (await self.execute(query)).results

    def close(self, *, wait: bool = True) -> None:
        close = getattr(self._service, "close", None)
        if callable(close):
            close(wait=wait)


__all__ = ["TRAKEModeAdapter", "TRAKEServicePort", "TRAKEDiagnostics"]
