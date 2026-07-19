from __future__ import annotations

import pytest

from online.domain.errors import InvalidQueryError
from query_understanding import TRAKEQueryBuilder, parse_trake_query


def test_builds_ordered_query_with_deterministic_event_ids() -> None:
    query = parse_trake_query(
        "trake-1",
        (
            "  Một người bước vào phòng  ",
            "Người đó   ngồi xuống",
            "Người đó rời đi",
        ),
        top_k_videos=3,
        policy={"lambda_penalty": 0.004},
    )

    assert query.query_id == "trake-1"
    assert tuple(event.event_id for event in query.events) == (
        "event-0001",
        "event-0002",
        "event-0003",
    )
    assert tuple(event.text for event in query.events) == (
        "Một người bước vào phòng",
        "Người đó ngồi xuống",
        "Người đó rời đi",
    )
    assert query.top_k_videos == 3
    assert query.policy.lambda_penalty == 0.004


def test_preserves_explicit_event_ids_and_order() -> None:
    query = TRAKEQueryBuilder().build(
        "trake-2",
        ("third", "first", "second"),
        event_ids=("e3", "e1", "e2"),
    )

    assert tuple((event.event_id, event.text) for event in query.events) == (
        ("e3", "third"),
        ("e1", "first"),
        ("e2", "second"),
    )


@pytest.mark.parametrize(
    ("descriptions", "event_ids"),
    [
        ((), None),
        (("one",), None),
        (("one", "  "), None),
        (("One event", "  one   event "), None),
        (("one", "two"), ("same", "same")),
        (("one", "two"), ("only-one",)),
    ],
)
def test_rejects_missing_duplicate_or_misaligned_events(
    descriptions: tuple[str, ...],
    event_ids: tuple[str, ...] | None,
) -> None:
    with pytest.raises(InvalidQueryError):
        parse_trake_query("trake-invalid", descriptions, event_ids=event_ids)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_id": " ", "event_descriptions": ("one", "two")},
        {"query_id": "q", "event_descriptions": "one then two"},
        {"query_id": "q", "event_descriptions": ("one", "two"), "top_k_videos": True},
        {
            "query_id": "q",
            "event_descriptions": ("one", "two"),
            "policy": {"lambda_penalty": float("nan")},
        },
    ],
)
def test_rejects_invalid_query_boundary(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidQueryError):
        parse_trake_query(**kwargs)  # type: ignore[arg-type]
