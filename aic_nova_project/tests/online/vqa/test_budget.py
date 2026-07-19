from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from online.vqa.budget import EvidenceBudgetPolicy


def test_defaults_match_dd_030_and_policy_is_frozen() -> None:
    policy = EvidenceBudgetPolicy()
    assert (policy.max_videos, policy.max_primary_per_video, policy.max_primary_total) == (3, 3, 8)
    assert policy.max_images_total == 12
    assert (policy.ocr_chars, policy.asr_chars) == (2_000, 4_000)
    assert (policy.summary_chars_per_video, policy.summary_chars_total) == (800, 2_400)
    assert policy.text_chars_total == 8_000
    assert policy.asr_window_seconds == 5.0
    with pytest.raises(FrozenInstanceError):
        policy.max_videos = 4  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, False])
def test_rejects_boolean(value: bool) -> None:
    with pytest.raises(TypeError):
        EvidenceBudgetPolicy(max_videos=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf])
def test_rejects_non_positive_or_non_finite(value: float) -> None:
    with pytest.raises(ValueError):
        EvidenceBudgetPolicy(asr_window_seconds=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_videos": 9},
        {"max_primary_per_video": 9},
        {"max_primary_total": 13},
        {"summary_chars_per_video": 2_401},
        {"summary_chars_total": 8_001},
        {"ocr_chars": 8_001},
    ],
)
def test_rejects_inconsistent_caps(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        EvidenceBudgetPolicy(**kwargs)
