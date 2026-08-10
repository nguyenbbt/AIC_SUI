"""Read-only search ports owned by Data & Infrastructure."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .records import ASRSearchHit, FrameSearchHit, VideoSearchHit


@runtime_checkable
class MilvusSearchPort(Protocol):
    def search_visual(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]: ...

    def search_ocr(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]: ...

    def search_asr(self, vector: Sequence[float], top_k: int) -> Sequence[ASRSearchHit]: ...

    def search_summary(self, vector: Sequence[float], top_k: int) -> Sequence[VideoSearchHit]: ...


@runtime_checkable
class ElasticsearchSearchPort(Protocol):
    def search_ocr(self, query: str, top_k: int, *, fuzzy: bool | None = None) -> Sequence[FrameSearchHit]: ...

    def search_asr(self, query: str, top_k: int, *, fuzzy: bool | None = None) -> Sequence[ASRSearchHit]: ...

    def search_summary(self, query: str, top_k: int, *, fuzzy: bool | None = None) -> Sequence[VideoSearchHit]: ...
