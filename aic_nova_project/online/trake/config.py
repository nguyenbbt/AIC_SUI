"""Validated configuration for the paper-aligned DANTE core."""

from __future__ import annotations

import math
from dataclasses import dataclass


DANTE_POLICY_NAME = "dante-index-gap-v1"
DEFAULT_DANTE_LAMBDA = 0.001
MIN_DANTE_LAMBDA = 0.001
MAX_DANTE_LAMBDA = 0.01


@dataclass(frozen=True)
class DANTEConfig:
    """Internal algorithm policy fixed by DD-026 through DD-029."""

    lambda_penalty: float = DEFAULT_DANTE_LAMBDA
    policy_name: str = DANTE_POLICY_NAME

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be a non-empty string")
        if (
            isinstance(self.lambda_penalty, bool)
            or not isinstance(self.lambda_penalty, (int, float))
            or not math.isfinite(float(self.lambda_penalty))
        ):
            raise ValueError("lambda_penalty must be a finite number")
        normalized = float(self.lambda_penalty)
        if not MIN_DANTE_LAMBDA <= normalized <= MAX_DANTE_LAMBDA:
            raise ValueError(
                f"lambda_penalty must be in [{MIN_DANTE_LAMBDA}, {MAX_DANTE_LAMBDA}]"
            )
        object.__setattr__(self, "lambda_penalty", normalized)
        object.__setattr__(self, "policy_name", self.policy_name.strip())


__all__ = [
    "DANTEConfig",
    "DANTE_POLICY_NAME",
    "DEFAULT_DANTE_LAMBDA",
    "MAX_DANTE_LAMBDA",
    "MIN_DANTE_LAMBDA",
]
