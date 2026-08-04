"""Versioned Vietnamese/English normalization for Open Images constraints."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType


OBJECT_LABEL_NORMALIZER_VERSION = "open_images_vi_en_v1"

_WHITESPACE = re.compile(r"\s+")
_DEFAULT_ALIASES = MappingProxyType(
    {
        "nguoi": "person",
        "người": "person",
        "nguoi dan ong": "man",
        "người đàn ông": "man",
        "nguoi phu nu": "woman",
        "người phụ nữ": "woman",
        "xe hoi": "car",
        "xe hơi": "car",
        "xe o to": "car",
        "xe ô tô": "car",
        "o to": "car",
        "ô tô": "car",
        "xe dap": "bicycle",
        "xe đạp": "bicycle",
        "dien thoai": "mobile phone",
        "điện thoại": "mobile phone",
        "dien thoai di dong": "mobile phone",
        "điện thoại di động": "mobile phone",
        "cell phone": "mobile phone",
    }
)


class ObjectLabelNormalizer:
    """Normalize user labels without changing A-owned query/domain models."""

    version = OBJECT_LABEL_NORMALIZER_VERSION

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        normalized_aliases = {
            self._normalize_text(source): self._normalize_text(target)
            for source, target in (aliases or _DEFAULT_ALIASES).items()
        }
        self._aliases = MappingProxyType(normalized_aliases)

    def normalize(self, label: str) -> str:
        normalized = self._normalize_text(label)
        return self._aliases.get(normalized, normalized)

    @staticmethod
    def _normalize_text(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("object label must be a non-empty string")
        normalized = unicodedata.normalize("NFKC", value).casefold().strip()
        return _WHITESPACE.sub(" ", normalized)


__all__ = [
    "OBJECT_LABEL_NORMALIZER_VERSION",
    "ObjectLabelNormalizer",
]
