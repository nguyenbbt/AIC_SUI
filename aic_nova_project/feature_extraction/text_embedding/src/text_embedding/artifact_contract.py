import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .encoders import BaseTextEncoder


TEXT_EMBEDDING_SCHEMA_VERSION = 1
_CANONICAL_FRAME_ID = re.compile(r"^.+_[0-9]{5}_[0-9]{3}$")


def source_sha256(source_path: Path) -> str:
    """Return a content fingerprint for one producer JSON artifact."""
    digest = hashlib.sha256()
    with source_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_encoder_provenance(
    encoder: BaseTextEncoder,
    artifact_kind: str,
) -> Dict[str, Any]:
    """Describe the encoder behavior that affects vector compatibility."""
    pooling_strategy = (
        "chunk_mean_l2"
        if artifact_kind == "summary"
        else "direct_l2"
    )
    model_revision = getattr(encoder, "model_revision", None)
    if (
        not isinstance(model_revision, str)
        or not model_revision.strip()
        or model_revision == "default"
    ):
        raise ValueError(
            "Encoder model_revision must be an explicit immutable revision"
        )
    return {
        "model_name": str(
            getattr(
                encoder,
                "model_name",
                f"{encoder.__class__.__module__}.{encoder.__class__.__name__}",
            )
        ),
        "model_revision": model_revision.strip(),
        "pooling_strategy": pooling_strategy,
        "max_length": int(getattr(encoder, "max_length", 0)),
        "embedding_dimension": int(
            getattr(encoder, "embedding_dim", 0)
        ),
        "normalized": True,
    }


def add_artifact_contract(
    records: List[Dict[str, Any]],
    *,
    artifact_kind: str,
    source_fingerprint: str,
    provenance: Dict[str, Any],
) -> None:
    """Attach resume/provenance fields to every Parquet row."""
    contract = {
        "artifact_schema_version": TEXT_EMBEDDING_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "source_sha256": source_fingerprint,
        **provenance,
    }
    for record in records:
        record.update(contract)


def _record_keys(
    records: Sequence[Dict[str, Any]],
    artifact_kind: str,
) -> List[str]:
    if artifact_kind == "asr":
        return [
            f"{record.get('video_id')}:{record.get('interval_id')}"
            for record in records
        ]
    if artifact_kind == "ocr":
        return [str(record.get("frame_id", "")) for record in records]
    return [str(record.get("video_id", "")) for record in records]


def is_valid_text_embedding_artifact(
    output_path: Path,
    *,
    expected_records: Sequence[Dict[str, Any]],
    artifact_kind: str,
    source_fingerprint: str,
    provenance: Dict[str, Any],
) -> bool:
    """Validate a text embedding artifact before resume skips encoding."""
    try:
        dataframe = pd.read_parquet(output_path)
    except Exception:
        return False

    if dataframe.empty or len(dataframe) != len(expected_records):
        return False

    required_columns = {
        "video_id",
        "text",
        "embedding",
        "artifact_schema_version",
        "artifact_kind",
        "source_sha256",
        *provenance.keys(),
    }
    if not required_columns.issubset(dataframe.columns):
        return False

    expected_contract = {
        "artifact_schema_version": TEXT_EMBEDDING_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "source_sha256": source_fingerprint,
        **provenance,
    }
    for column, expected_value in expected_contract.items():
        if not all(
            value == expected_value
            for value in dataframe[column].tolist()
        ):
            return False

    actual_records = dataframe.to_dict(orient="records")
    if _record_keys(actual_records, artifact_kind) != _record_keys(
        expected_records,
        artifact_kind,
    ):
        return False
    if dataframe["text"].tolist() != [
        record["text"]
        for record in expected_records
    ]:
        return False

    if artifact_kind == "asr":
        interval_ids = dataframe["interval_id"].tolist()
        if any(
            not isinstance(interval_id, str)
            or not interval_id.isascii()
            or not interval_id.isdigit()
            for interval_id in interval_ids
        ):
            return False
    elif artifact_kind == "ocr":
        if any(
            not _CANONICAL_FRAME_ID.fullmatch(frame_id)
            for frame_id in dataframe["frame_id"].tolist()
        ):
            return False

    expected_dimension = provenance["embedding_dimension"]
    if expected_dimension <= 0:
        return False
    for embedding in dataframe["embedding"].tolist():
        try:
            vector = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError):
            return False
        if vector.shape != (expected_dimension,):
            return False
        if not np.isfinite(vector).all():
            return False
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3):
            return False

    return True
