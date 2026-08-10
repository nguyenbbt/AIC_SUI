"""Interface-only image resolution boundary; no real path policy is defined."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from online.domain.vqa import ImageEvidence


@runtime_checkable
class ImageResolverPort(Protocol):
    def resolve_images(self, frame_ids: Sequence[str]) -> Mapping[str, ImageEvidence]: ...
