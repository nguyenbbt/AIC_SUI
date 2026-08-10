"""Read-only manifest loader and startup identity gate."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import ValidationError

from online.config import DatasetResourceConfig
from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.domain.manifest import DatasetManifest


class DatasetManifestGate:
    """Lock one Online process to exactly one READY Offline dataset identity."""

    def __init__(self, config: DatasetResourceConfig) -> None:
        self.config = config
        self._manifest: DatasetManifest | None = None
        self._lock = threading.RLock()

    @property
    def manifest(self) -> DatasetManifest:
        with self._lock:
            if self._manifest is None:
                raise ResourceUnavailableError("Dataset manifest gate is not connected")
            return self._manifest

    def connect(self) -> None:
        with self._lock:
            if self._manifest is not None:
                return
            self._manifest = self._load_and_validate()

    def close(self) -> None:
        with self._lock:
            self._manifest = None

    def health_check(self) -> None:
        with self._lock:
            active = self._manifest
            if active is None:
                raise ResourceUnavailableError("Dataset manifest gate is not connected")
            current = self._load_and_validate()
            if (
                current.dataset_id != active.dataset_id
                or current.dataset_fingerprint != active.dataset_fingerprint
            ):
                raise ContractMismatchError(
                    "Active dataset manifest changed after startup; restart is required"
                )

    def _load_and_validate(self) -> DatasetManifest:
        path = Path(self.config.manifest_path).expanduser().resolve()
        if not path.is_file():
            raise ResourceUnavailableError(
                "Dataset manifest does not exist",
                details={"resource": "dataset_manifest"},
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ResourceUnavailableError(
                "Dataset manifest cannot be read",
                details={"resource": "dataset_manifest"},
            ) from exc
        try:
            manifest = DatasetManifest.model_validate(payload)
        except ValidationError as exc:
            raise ContractMismatchError(
                "Dataset manifest violates self-indexed-v2",
                details={"resource": "dataset_manifest"},
            ) from exc
        if manifest.contract_version != self.config.expected_contract_version:
            raise ContractMismatchError("Dataset contract version is not the deployed version")
        expected = self.config.expected_fingerprint
        if expected is not None and manifest.dataset_fingerprint != expected:
            raise ContractMismatchError("Dataset fingerprint does not match deployment")
        return manifest


__all__ = ["DatasetManifestGate"]
