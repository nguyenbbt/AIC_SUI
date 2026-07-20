"""Validated parser for explicitly ordered TRAKE event descriptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from online.domain.errors import InvalidQueryError
from online.domain.trake import DANTEPolicy, TRAKEEvent, TRAKEQuery


class TRAKEQueryBuilder:
    """Build one TRAKE query without KIS variants or heuristic event splitting."""

    def build(
        self,
        query_id: str,
        event_descriptions: Sequence[str],
        *,
        event_ids: Sequence[str] | None = None,
        top_k_videos: int = 1,
        policy: DANTEPolicy | Mapping[str, object] | None = None,
    ) -> TRAKEQuery:
        normalized_query_id = self._clean_text(query_id, "query_id")
        if isinstance(event_descriptions, (str, bytes)):
            raise InvalidQueryError(
                "event_descriptions must be an ordered sequence of text values"
            )
        try:
            raw_descriptions = tuple(event_descriptions)
        except TypeError as exc:
            raise InvalidQueryError(
                "event_descriptions must be an ordered sequence of text values"
            ) from exc

        descriptions = tuple(
            self._normalize_event_text(value, index)
            for index, value in enumerate(raw_descriptions)
        )
        if len(descriptions) < 2:
            raise InvalidQueryError("TRAKE requires at least two ordered events")
        duplicate_keys = tuple(description.casefold() for description in descriptions)
        if len(set(duplicate_keys)) != len(duplicate_keys):
            raise InvalidQueryError("TRAKE event descriptions must be unique")

        normalized_event_ids = self._event_ids(event_ids, len(descriptions))
        try:
            normalized_policy = (
                DANTEPolicy()
                if policy is None
                else policy
                if isinstance(policy, DANTEPolicy)
                else DANTEPolicy.model_validate(policy)
            )
            events = tuple(
                TRAKEEvent(event_id=event_id, text=description)
                for event_id, description in zip(
                    normalized_event_ids,
                    descriptions,
                    strict=True,
                )
            )
            return TRAKEQuery(
                query_id=normalized_query_id,
                events=events,
                top_k_videos=top_k_videos,
                policy=normalized_policy,
            )
        except ValidationError as exc:
            raise InvalidQueryError(
                "Invalid TRAKE query",
                details={"validation_error_count": exc.error_count()},
            ) from exc

    @classmethod
    def _event_ids(
        cls,
        event_ids: Sequence[str] | None,
        event_count: int,
    ) -> tuple[str, ...]:
        if event_ids is None:
            return tuple(f"event-{index:04d}" for index in range(1, event_count + 1))
        if isinstance(event_ids, (str, bytes)):
            raise InvalidQueryError("event_ids must be an ordered sequence")
        try:
            values = tuple(event_ids)
        except TypeError as exc:
            raise InvalidQueryError("event_ids must be an ordered sequence") from exc
        if len(values) != event_count:
            raise InvalidQueryError(
                "event_ids count must match event_descriptions count",
                details={"expected": event_count, "actual": len(values)},
            )
        normalized = tuple(
            cls._clean_text(value, f"event_ids[{index}]")
            for index, value in enumerate(values)
        )
        if len(set(normalized)) != len(normalized):
            raise InvalidQueryError("TRAKE event IDs must be unique")
        return normalized

    @classmethod
    def _normalize_event_text(cls, value: object, index: int) -> str:
        cleaned = cls._clean_text(value, f"event_descriptions[{index}]")
        return " ".join(cleaned.split())

    @staticmethod
    def _clean_text(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidQueryError(f"{name} must be non-empty text")
        return value.strip()


_TRAKE_BUILDER = TRAKEQueryBuilder()


def parse_trake_query(
    query_id: str,
    event_descriptions: Sequence[str],
    *,
    event_ids: Sequence[str] | None = None,
    top_k_videos: int = 1,
    policy: DANTEPolicy | Mapping[str, object] | None = None,
) -> TRAKEQuery:
    """Public facade for already-separated, explicitly ordered TRAKE events."""

    return _TRAKE_BUILDER.build(
        query_id,
        event_descriptions,
        event_ids=event_ids,
        top_k_videos=top_k_videos,
        policy=policy,
    )


__all__ = ["TRAKEQueryBuilder", "parse_trake_query"]
