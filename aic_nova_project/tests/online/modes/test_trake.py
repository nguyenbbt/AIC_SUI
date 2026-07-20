from __future__ import annotations

import asyncio

import pytest

from online.domain.errors import ContractMismatchError
from online.domain.trake import DANTEPolicy, TRAKEDiagnostics, TRAKEEvent, TRAKEQuery
from online.modes.trake import TRAKEModeAdapter
from online.trake.service import TRAKEExecution


def _query() -> TRAKEQuery:
    return TRAKEQuery(
        query_id="trake-1",
        events=(TRAKEEvent(event_id="e1", text="first"), TRAKEEvent(event_id="e2", text="second")),
    )


def _diagnostics() -> TRAKEDiagnostics:
    return TRAKEDiagnostics(
        policy_version=DANTEPolicy().policy_version,
        lambda_penalty=DANTEPolicy().lambda_penalty,
        event_count=2,
        video_count=0,
        frame_count=0,
    )


class _Service:
    def __init__(self) -> None:
        self.calls = []

    async def execute(self, query):
        self.calls.append(query)
        return TRAKEExecution(query_id=query.query_id, results=(), diagnostics=_diagnostics())


def test_trake_mode_delegates_once_and_preserves_query_id() -> None:
    service = _Service()
    result = asyncio.run(TRAKEModeAdapter(service).execute(_query()))
    assert result.query_id == "trake-1"
    assert service.calls == [_query()]


def test_trake_mode_rejects_non_domain_input() -> None:
    with pytest.raises(ContractMismatchError):
        asyncio.run(TRAKEModeAdapter(_Service()).execute(object()))
