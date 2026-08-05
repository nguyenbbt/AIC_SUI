import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def write_embeddings_to_parquet(
    records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Writes a list of records (including 'embedding' as a flat array/list) to a Parquet file.
    """
    if not records:
        logger.warning(f"No records to write for {output_path}")
        return
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        df = pd.DataFrame(records)
        df.to_parquet(temp_path, engine="pyarrow", index=False)
        temp_path.replace(output_path)
        logger.info(f"Successfully saved {len(records)} embeddings to {output_path}")
    except Exception:
        logger.exception("Failed to write Parquet to %s", output_path)
        raise
    finally:
        temp_path.unlink(missing_ok=True)
