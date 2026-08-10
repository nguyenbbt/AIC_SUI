"""Frozen VQA evidence-budget policy from DD-030."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, slots=True)
class EvidenceBudgetPolicy:
    """Limits applied by the pure VQA evidence selector.

    Defaults live only here so the Wave 2 adapter can map the public budget
    contract to one policy object without duplicating policy values.
    """

    max_videos: int = 3
    max_primary_per_video: int = 3
    max_primary_total: int = 8
    max_images_total: int = 12
    ocr_chars: int = 2_000
    asr_chars: int = 4_000
    summary_chars_per_video: int = 800
    summary_chars_total: int = 2_400
    text_chars_total: int = 8_000
    asr_window_seconds: float = 5.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field.name} must be a positive finite number")
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if value <= 0:
                raise ValueError(f"{field.name} must be greater than zero")

        integer_fields = (
            "max_videos",
            "max_primary_per_video",
            "max_primary_total",
            "max_images_total",
            "ocr_chars",
            "asr_chars",
            "summary_chars_per_video",
            "summary_chars_total",
            "text_chars_total",
        )
        for name in integer_fields:
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")

        if self.max_videos > self.max_primary_total:
            raise ValueError("max_videos must not exceed max_primary_total")
        if self.max_primary_per_video > self.max_primary_total:
            raise ValueError("max_primary_per_video must not exceed max_primary_total")
        if self.max_primary_total > self.max_images_total:
            raise ValueError("max_primary_total must not exceed max_images_total")
        if self.summary_chars_per_video > self.summary_chars_total:
            raise ValueError("summary_chars_per_video must not exceed summary_chars_total")
        if self.summary_chars_total > self.text_chars_total:
            raise ValueError("summary_chars_total must not exceed text_chars_total")
        if max(self.ocr_chars, self.asr_chars) > self.text_chars_total:
            raise ValueError("individual text caps must not exceed text_chars_total")
