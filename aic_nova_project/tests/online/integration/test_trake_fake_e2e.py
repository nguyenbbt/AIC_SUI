from __future__ import annotations

import asyncio

from online.testing import build_advanced_modes_fixture
from online.trake import TRAKEService


def test_trake_shared_fixture_runs_end_to_end_deterministically() -> None:
    fixture = build_advanced_modes_fixture()
    encoder = fixture.text_encoder()
    corpus = fixture.visual_corpus()

    async def scenario():
        service = TRAKEService(corpus=corpus, encoder=encoder)
        try:
            return await service.execute(fixture.trake_query)
        finally:
            service.close()

    execution = asyncio.run(scenario())

    assert tuple(result.video_id for result in execution.results) == (
        "L21_V001",
        "L21_V003",
        "L21_V002",
    )
    winner = execution.results[0]
    assert winner.video_id == fixture.expected_dante_video_id
    assert winner.score == fixture.expected_dante_score
    assert tuple(match.local_index for match in winner.sequence) == (
        fixture.expected_dante_positions
    )
    tie_result = next(
        result
        for result in execution.results
        if result.video_id == fixture.tie_video_id
    )
    assert tuple(match.local_index for match in tie_result.sequence) == (
        fixture.tied_sequence_positions[0]
    )
    assert execution.diagnostics.invalid_sequence_count == 1
    assert len(encoder.calls) == 1
