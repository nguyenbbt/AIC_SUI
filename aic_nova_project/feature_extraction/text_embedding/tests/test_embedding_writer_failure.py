from pathlib import Path
from unittest.mock import patch

import pytest

from src.text_embedding.embedding_writer import (
    write_embeddings_to_parquet,
)


def test_parquet_write_failure_is_propagated_and_preserves_old_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "V001.parquet"
    output_path.write_bytes(b"last-known-good")
    records = [{"frame_id": "V001_00000_015", "embedding": [1.0, 0.0]}]

    with patch(
        "src.text_embedding.embedding_writer.pd.DataFrame.to_parquet",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            write_embeddings_to_parquet(records, output_path)

    assert output_path.read_bytes() == b"last-known-good"
