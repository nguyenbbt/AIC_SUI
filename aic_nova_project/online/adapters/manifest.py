"""Read-only JSON adapter for the organizer ingestion manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from online.config import ManifestResourceConfig
from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.ports.manifest import DatasetManifest


class JsonManifestAdapter:
    _FIELDS = tuple(DatasetManifest.model_fields)

    def __init__(self, config: ManifestResourceConfig) -> None:
        self.config = config

    def read_manifest(self) -> DatasetManifest:
        path = Path(self.config.path).expanduser()
        if not path.is_file():
            raise ResourceUnavailableError(
                "Organizer ingestion manifest is unavailable",
                details={"resource": "dataset_manifest"},
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractMismatchError(
                "Organizer ingestion manifest could not be parsed",
                details={"resource": "dataset_manifest"},
            ) from exc
        if not isinstance(payload, Mapping):
            raise ContractMismatchError(
                "Organizer ingestion manifest must be a JSON object",
                details={"resource": "dataset_manifest"},
            )
        selected = {field: payload.get(field) for field in self._FIELDS}
        try:
            return DatasetManifest.model_validate(selected)
        except ValidationError as exc:
            raise ContractMismatchError(
                "Organizer ingestion manifest fields are invalid",
                details={"resource": "dataset_manifest"},
            ) from exc


__all__ = ["JsonManifestAdapter"]
