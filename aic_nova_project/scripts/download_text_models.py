import os
import argparse
from sentence_transformers import SentenceTransformer

def download_model(
    model_name: str,
    cache_dir: str,
    model_revision: str,
):
    """
    Downloads the model to the specified cache directory.
    This ensures that the model is cached for offline use.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    print(f"Downloading model '{model_name}' to '{cache_dir}'...")
    # This will download and cache the model
    SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        revision=model_revision,
    )
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="dangvantuan/vietnamese-embedding")
    parser.add_argument("--model-revision", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="/app/models")
    args = parser.parse_args()
    
    download_model(args.model_name, args.cache_dir, args.model_revision)
