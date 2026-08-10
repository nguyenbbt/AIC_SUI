"""Encoder ports; implementations belong to Query & Retrieval."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class TextEncoderPort(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class ImageEncoderPort(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_images(self, images: Sequence[object]) -> Sequence[Sequence[float]]: ...
