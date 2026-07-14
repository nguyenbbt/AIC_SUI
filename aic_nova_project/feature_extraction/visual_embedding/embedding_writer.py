import os
import pandas as pd
from typing import List, Dict, Any

def write_embeddings_to_parquet(
    video_id: str,
    records: List[Dict[str, Any]], 
    output_dir: str
):
    """
    Writes the embeddings and metadata for a single video to a Parquet file.
    
    Args:
        video_id: ID of the video.
        records: List of dictionaries containing frame_id, video_id, shot_id, position, 
                 model_name, embedding_dim, and embedding.
        output_dir: Directory to save the Parquet file.
    """
    if not records:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(output_dir, f"{video_id}.parquet")
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(records)
    
    # Ensure embedding is saved as a list of floats
    # df['embedding'] is already a list (or numpy array).
    # pyarrow can handle lists/numpy arrays. We'll ensure it's a list for compatibility.
    df['embedding'] = df['embedding'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    # Write to Parquet using PyArrow backend
    df.to_parquet(output_path, engine='pyarrow', index=False)
