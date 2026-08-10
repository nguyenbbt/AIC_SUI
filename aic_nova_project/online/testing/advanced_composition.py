"""Testing-only adapter from the advanced fake bundle to production wiring."""

from __future__ import annotations

from collections.abc import Callable

from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    ResourceUnavailableError,
)
from online.testing.advanced_runtime import AdvancedRuntimeBundle, AdvancedRuntimeState
from retrieval_api.composition import (
    AdvancedModeDependencies,
    OnlineRuntime,
    attach_advanced_modes,
)


def attach_advanced_fake_modes(
    runtime: OnlineRuntime,
    bundle: AdvancedRuntimeBundle,
) -> OnlineRuntime:
    dependencies = AdvancedModeDependencies(
        visual_corpus=bundle.visual_corpus,
        event_encoder=bundle.text_encoder,
        metadata_reader=bundle.metadata_reader,
        image_resolver=bundle.image_resolver,
        evidence_hydrator=bundle.evidence_hydrator,
        vlm=bundle.vlm,
        managed_resources=(bundle,),
    )
    return attach_advanced_modes(
        runtime,
        dependencies=dependencies,
        trake_readiness=_fake_readiness(bundle, mode="trake"),
        vqa_readiness=_fake_readiness(bundle, mode="vqa"),
    )


def _fake_readiness(bundle: AdvancedRuntimeBundle, *, mode: str) -> Callable[[], None]:
    if mode == "trake":
        components = (
            ("event_encoder", bundle.config.encoder_state),
            ("visual_corpus", bundle.config.visual_state),
        )
    else:
        components = (
            ("metadata_reader", bundle.config.metadata_state),
            ("ocr_evidence", bundle.config.ocr_state),
            ("asr_evidence", bundle.config.asr_state),
            ("summary_evidence", bundle.config.summary_state),
            ("image_resolver", bundle.config.image_state),
            ("vlm", bundle.config.vlm_state),
        )

    def check() -> None:
        if bundle.closed:
            raise ResourceUnavailableError(f"{mode} fake runtime is closed")
        for component, state in components:
            details = {"component": component}
            if state is AdvancedRuntimeState.TIMEOUT:
                raise BranchTimeoutError(f"{mode} readiness probe timed out", details=details)
            if state is AdvancedRuntimeState.UNAVAILABLE:
                raise ResourceUnavailableError(f"{mode} resource is unavailable", details=details)
            if state is AdvancedRuntimeState.INVALID_REFERENCE:
                raise ContractMismatchError(f"{mode} resource has an invalid reference", details=details)

    return check


__all__ = ["attach_advanced_fake_modes"]
