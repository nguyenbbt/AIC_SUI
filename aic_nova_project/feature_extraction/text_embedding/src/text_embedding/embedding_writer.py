import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def write_embeddings_to_parquet(records: List[Dict[str, Any]], output_path: Path):
    """
    Writes a list of records (including 'embedding' as a flat array/list) to a Parquet file.
    """
    if not records:
        logger.warning(f"No records to write for {output_path}")
        return
        
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Successfully saved {len(records)} embeddings to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write Parquet to {output_path}: {e}")
