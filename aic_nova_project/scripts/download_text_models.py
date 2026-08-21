import os
import argparse
from huggingface_hub import snapshot_download

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
    snapshot_download(
        repo_id=model_name,
        revision=model_revision,
        cache_dir=cache_dir,
    )
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="dangvantuan/vietnamese-embedding")
    parser.add_argument("--model-revision", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="/app/models")
    args = parser.parse_args()
    
    download_model(args.model_name, args.cache_dir, args.model_revision)
