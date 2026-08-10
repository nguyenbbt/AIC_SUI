import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_VISUAL_MODEL_ID,
    EXPECTED_VISUAL_EMBEDDING_DIMENSION,
)


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "frame_id",
    "model_id",
    "precision",
    "source_size_bytes",
    "source_mtime_ns",
    "embedding_dim",
    "embedding",
}


def visual_output_is_valid(
    output_path: str | Path,
    expected_records: List[Dict[str, Any]],
    model_id: str,
    precision: str,
) -> bool:
    """Validate whether a visual Parquet artifact is safe to resume."""
    try:
        dataframe = pd.read_parquet(output_path)
        if not REQUIRED_COLUMNS.issubset(dataframe.columns):
            return False
        if len(dataframe) != len(expected_records):
            return False
        if dataframe["frame_id"].duplicated().any():
            return False

        expected_by_id = {
            str(record["frame_id"]): record for record in expected_records
        }
        if set(dataframe["frame_id"].astype(str)) != set(expected_by_id):
            return False
        if not (dataframe["model_id"].astype(str) == model_id).all():
            return False
        if not (dataframe["precision"].astype(str) == precision).all():
            return False

        for _, row in dataframe.iterrows():
            frame_id = str(row["frame_id"])
            source_path = Path(expected_by_id[frame_id]["file_path"])
            if not source_path.is_file():
                return False

            source_stat = source_path.stat()
            if int(row["source_size_bytes"]) != source_stat.st_size:
                return False
            if int(row["source_mtime_ns"]) != source_stat.st_mtime_ns:
                return False

            embedding = np.asarray(row["embedding"], dtype=np.float32)
            embedding_dim = int(row["embedding_dim"])
            if (
                model_id == DEFAULT_VISUAL_MODEL_ID
                and embedding_dim != EXPECTED_VISUAL_EMBEDDING_DIMENSION
            ):
                return False
            if embedding.ndim != 1 or embedding.size != embedding_dim:
                return False
            if embedding_dim <= 0 or not np.isfinite(embedding).all():
                return False
            if not np.isclose(
                np.linalg.norm(embedding),
                1.0,
                rtol=1e-3,
                atol=1e-3,
            ):
                return False

        return True
    except Exception as exc:
        logger.warning(
            "Visual resume validation failed for %s: %s",
            output_path,
            exc,
        )
        return False
