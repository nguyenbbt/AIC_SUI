"""Object constraints inspired by paper object-filtering stages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from online.domain.candidates import CandidateDiagnostics, FusedFrameCandidate
from online.domain.enums import CountOperator, FilterMode
from online.domain.errors import InvalidQueryError
from online.domain.query import ObjectConstraint
from online.ports.objects import ObjectReaderPort


@dataclass(frozen=True)
class ObjectProcessingConfig:
    soft_boost_per_constraint: float = 0.05
    max_total_boost: float = 0.2

    def __post_init__(self) -> None:
        if not _finite_non_negative(self.soft_boost_per_constraint):
            raise ValueError("soft_boost_per_constraint must be finite and >= 0")
        if not _finite_non_negative(self.max_total_boost):
            raise ValueError("max_total_boost must be finite and >= 0")


class ObjectConstraintProcessor:
    """Apply hard filters and soft boosts from structured object constraints."""

    name = "object_constraints"

    def __init__(
        self,
        object_reader: ObjectReaderPort,
        *,
        config: ObjectProcessingConfig | None = None,
    ) -> None:
        if not isinstance(object_reader, ObjectReaderPort):
            raise TypeError("object_reader must implement ObjectReaderPort")
        self.object_reader = object_reader
        self.config = config or ObjectProcessingConfig()

    def process(
        self,
        candidates: Sequence[FusedFrameCandidate],
        constraints: Sequence[ObjectConstraint],
    ) -> tuple[FusedFrameCandidate, ...]:
        values = tuple(candidates)
        constraint_values = tuple(constraints)
        if not values or not constraint_values:
            return values
        if any(constraint.position is not None for constraint in constraint_values):
            raise InvalidQueryError(
                "object position constraints are not supported by the current online contract",
                details={"constraint": "object_position"},
            )

        frame_ids = tuple(candidate.frame_id for candidate in values)
        objects_by_frame = self.object_reader.get_objects_by_frame_ids(frame_ids)
        output: list[FusedFrameCandidate] = []
        for candidate in values:
            objects = tuple(objects_by_frame.get(candidate.frame_id, ()))
            satisfied = tuple(
                constraint
                for constraint in constraint_values
                if _constraint_satisfied(constraint, objects)
            )
            if any(
                constraint.filter_mode is FilterMode.HARD
                and constraint not in satisfied
                for constraint in constraint_values
            ):
                continue
            soft_matches = sum(
                1
                for constraint in satisfied
                if constraint.filter_mode is FilterMode.SOFT
            )
            boost = min(
                self.config.max_total_boost,
                soft_matches * self.config.soft_boost_per_constraint,
            )
            diagnostics = CandidateDiagnostics(
                summary_boost=candidate.diagnostics.summary_boost,
                object_boost=candidate.diagnostics.object_boost + boost,
                object_constraints_satisfied=len(satisfied),
            )
            output.append(
                candidate.model_copy(
                    update={
                        "final_score": candidate.final_score + boost,
                        "objects": objects,
                        "diagnostics": diagnostics,
                    }
                )
            )
        return tuple(sorted(output, key=lambda item: (-item.final_score, item.frame_id)))


def _constraint_satisfied(
    constraint: ObjectConstraint,
    objects: Sequence[object],
) -> bool:
    count = sum(
        1
        for obj in objects
        if getattr(obj, "label", None) == constraint.label
        and getattr(obj, "confidence", -1.0) >= constraint.min_confidence
    )
    if constraint.count_operator is CountOperator.EQ:
        return count == constraint.count
    if constraint.count_operator is CountOperator.GTE:
        return count >= constraint.count
    if constraint.count_operator is CountOperator.LTE:
        return count <= constraint.count
    raise ValueError(f"unsupported count operator: {constraint.count_operator}")


def _finite_non_negative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
