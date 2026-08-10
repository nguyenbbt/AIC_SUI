from __future__ import annotations

import math
import random
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from online.domain.trake import (
    DANTE_DEFAULT_LAMBDA as PUBLIC_DANTE_DEFAULT_LAMBDA,
    DANTE_MAX_LAMBDA as PUBLIC_DANTE_MAX_LAMBDA,
    DANTE_MIN_LAMBDA as PUBLIC_DANTE_MIN_LAMBDA,
    DANTE_POLICY_VERSION,
)
from online.trake import dante as dante_module
from online.trake import (
    DANTEConfig,
    DANTEPath,
    DANTE_POLICY_NAME,
    DEFAULT_DANTE_LAMBDA,
    MAX_DANTE_LAMBDA,
    MIN_DANTE_LAMBDA,
    solve_dante,
)


def _solve_naively(
    similarities: tuple[tuple[float, ...], ...],
    lambda_penalty: float,
) -> tuple[float, tuple[int, ...]] | None:
    """Small O(N*T^2) oracle for checking the optimized recurrence."""

    event_count = len(similarities)
    frame_count = len(similarities[0])
    if frame_count < event_count:
        return None

    scores = [[-math.inf] * frame_count for _ in range(event_count)]
    backpointers = [[-1] * frame_count for _ in range(event_count)]
    scores[0] = list(similarities[0])

    for event_index in range(1, event_count):
        for frame_position in range(frame_count):
            best_score = -math.inf
            best_predecessor = -1
            for predecessor in range(frame_position):
                if not math.isfinite(scores[event_index - 1][predecessor]):
                    continue
                candidate = (
                    scores[event_index - 1][predecessor]
                    - lambda_penalty * (frame_position - predecessor)
                )
                if candidate > best_score or (
                    candidate == best_score
                    and (
                        best_predecessor < 0
                        or predecessor < best_predecessor
                    )
                ):
                    best_score = candidate
                    best_predecessor = predecessor

            if best_predecessor >= 0:
                scores[event_index][frame_position] = (
                    similarities[event_index][frame_position] + best_score
                )
                backpointers[event_index][frame_position] = best_predecessor

    final_position = -1
    final_score = -math.inf
    for position, score in enumerate(scores[-1]):
        if score > final_score:
            final_position = position
            final_score = score
    if final_position < 0:
        return None

    positions = [final_position]
    for event_index in range(event_count - 1, 0, -1):
        positions.append(backpointers[event_index][positions[-1]])
    positions.reverse()
    return final_score, tuple(positions)


def test_config_defaults_match_decisions() -> None:
    config = DANTEConfig()

    assert config.lambda_penalty == DEFAULT_DANTE_LAMBDA == 0.001
    assert config.policy_name == DANTE_POLICY_NAME
    assert MIN_DANTE_LAMBDA == 0.001
    assert MAX_DANTE_LAMBDA == 0.01


def test_internal_config_matches_public_wave_one_contract() -> None:
    assert DANTE_POLICY_NAME == DANTE_POLICY_VERSION
    assert DEFAULT_DANTE_LAMBDA == PUBLIC_DANTE_DEFAULT_LAMBDA
    assert MIN_DANTE_LAMBDA == PUBLIC_DANTE_MIN_LAMBDA
    assert MAX_DANTE_LAMBDA == PUBLIC_DANTE_MAX_LAMBDA


@pytest.mark.parametrize("lambda_penalty", [0.001, 0.004, 0.01])
def test_config_accepts_and_normalizes_supported_numeric_values(
    lambda_penalty: float,
) -> None:
    config = DANTEConfig(lambda_penalty=lambda_penalty, policy_name=" custom ")

    assert isinstance(config.lambda_penalty, float)
    assert config.lambda_penalty == lambda_penalty
    assert config.policy_name == "custom"


@pytest.mark.parametrize(
    "lambda_penalty",
    [True, "0.001", math.nan, math.inf, -math.inf, 0.0009, 0.0101],
)
def test_config_rejects_invalid_lambda(lambda_penalty: object) -> None:
    with pytest.raises(ValueError):
        DANTEConfig(lambda_penalty=lambda_penalty)  # type: ignore[arg-type]


@pytest.mark.parametrize("policy_name", ["", "   ", None, 7])
def test_config_rejects_invalid_policy_name(policy_name: object) -> None:
    with pytest.raises(ValueError, match="policy_name"):
        DANTEConfig(policy_name=policy_name)  # type: ignore[arg-type]


def test_config_is_frozen() -> None:
    config = DANTEConfig()

    with pytest.raises(FrozenInstanceError):
        config.lambda_penalty = 0.004  # type: ignore[misc]


def test_solver_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="DANTEConfig"):
        solve_dante(((0.5,),), config=0)  # type: ignore[arg-type]


def test_solves_known_two_event_path() -> None:
    result = solve_dante(
        (
            (0.9, 0.2, 0.1),
            (0.1, 0.4, 0.95),
        )
    )

    assert result is not None
    assert result.positions == (0, 2)
    assert result.score == pytest.approx(1.848)


def test_solves_known_three_event_path() -> None:
    result = solve_dante(
        (
            (0.9, 0.1, 0.0, 0.0, 0.0),
            (0.0, 0.8, 0.1, 0.0, 0.0),
            (0.0, 0.0, 0.7, 0.1, 0.0),
        )
    )

    assert result is not None
    assert result.positions == (0, 1, 2)
    assert result.score == pytest.approx(2.398)


def test_supports_one_event_and_prefers_smaller_final_position_on_tie() -> None:
    result = solve_dante(((0.1, 0.8, 0.8, -0.2),))

    assert result == DANTEPath(score=0.8, positions=(1,))


def test_returns_none_when_video_has_fewer_frames_than_events() -> None:
    assert solve_dante(((0.2,), (0.8,))) is None


def test_requires_strictly_later_frame_for_each_event() -> None:
    result = solve_dante(
        (
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
        )
    )

    assert result is not None
    assert result.positions == (0, 1)
    assert result.positions[0] < result.positions[1]


def test_handles_negative_cosine_similarities() -> None:
    matrix = (
        (-0.9, -0.2, -0.7),
        (-0.8, -0.6, -0.1),
    )

    result = solve_dante(matrix)
    expected = _solve_naively(matrix, DEFAULT_DANTE_LAMBDA)

    assert result is not None
    assert expected is not None
    assert result.positions == expected[1]
    assert result.score == pytest.approx(expected[0])


def test_prefers_smaller_predecessor_position_on_exact_tie() -> None:
    result = solve_dante(
        (
            (0.5, 0.499, 0.0),
            (-1.0, -1.0, 0.5),
        )
    )

    assert result is not None
    assert result.positions == (0, 2)


def test_prefers_smaller_final_position_on_exact_tie() -> None:
    result = solve_dante(
        (
            (0.0, -1.0, 0.0),
            (-1.0, 0.0, 0.001),
        )
    )

    assert result is not None
    assert result.positions == (0, 1)
    assert result.score == pytest.approx(-0.001)


def test_result_normalizes_positions_to_immutable_tuple() -> None:
    result = DANTEPath(score=1, positions=[0, 2])  # type: ignore[arg-type]

    assert result.score == 1.0
    assert result.positions == (0, 2)


@pytest.mark.parametrize(
    ("score", "positions", "error_type"),
    [
        (True, (0,), TypeError),
        ("1", (0,), TypeError),
        (math.nan, (0,), ValueError),
        (math.inf, (0,), ValueError),
        (1.0, (), ValueError),
        (1.0, (-1,), ValueError),
        (1.0, (True,), ValueError),
        (1.0, (0, 0), ValueError),
        (1.0, (2, 1), ValueError),
    ],
)
def test_result_rejects_invalid_values(
    score: object,
    positions: tuple[object, ...],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        DANTEPath(score=score, positions=positions)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("matrix", "error_type", "message"),
    [
        ([], ValueError, "at least one"),
        ([[]], ValueError, "must not be empty"),
        ([[0.1], [0.2, 0.3]], ValueError, "equal length"),
        ("bad", TypeError, "sequence of rows"),
        (["bad"], TypeError, "row must be a sequence"),
        ([[True]], ValueError, "finite numbers"),
        ([["0.2"]], ValueError, "finite numbers"),
        ([[math.nan]], ValueError, "finite numbers"),
        ([[math.inf]], ValueError, "finite numbers"),
        ([[1.01]], ValueError, r"in \[-1, 1\]"),
        ([[-1.01]], ValueError, r"in \[-1, 1\]"),
    ],
)
def test_rejects_invalid_similarity_matrices(
    matrix: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        solve_dante(matrix)  # type: ignore[arg-type]


def test_optimized_solver_matches_naive_oracle_on_random_matrices() -> None:
    random_generator = random.Random(20260718)

    for event_count in range(1, 6):
        for frame_count in range(1, 9):
            for lambda_penalty in (0.001, 0.004, 0.01):
                matrix = tuple(
                    tuple(
                        round(random_generator.uniform(-1.0, 1.0), 6)
                        for _ in range(frame_count)
                    )
                    for _ in range(event_count)
                )
                expected = _solve_naively(matrix, lambda_penalty)
                actual = solve_dante(
                    matrix,
                    DANTEConfig(lambda_penalty=lambda_penalty),
                )

                if expected is None:
                    assert actual is None
                else:
                    assert actual is not None
                    assert actual.positions == expected[1]
                    assert all(
                        current > previous
                        for previous, current in zip(
                            actual.positions, actual.positions[1:]
                        )
                    )
                    assert actual.score == pytest.approx(expected[0], abs=1e-12)


def test_production_recurrence_performs_only_linear_predecessor_updates() -> None:
    event_count = 10
    frame_count = 200
    matrix = tuple(
        tuple(0.1 for _ in range(frame_count))
        for _ in range(event_count)
    )

    with patch(
        "online.trake.dante._is_better_predecessor",
        wraps=dante_module._is_better_predecessor,
    ) as predecessor_comparison:
        result = solve_dante(matrix)

    assert result is not None
    assert predecessor_comparison.call_count <= event_count * frame_count
