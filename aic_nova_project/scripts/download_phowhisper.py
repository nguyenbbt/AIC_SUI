import os
import argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Download PhoWhisper models for offline use.")
    parser.add_argument("--size", type=str, default="medium", choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    model_name = f"vinai/PhoWhisper-{args.size}"
    print(f"Downloading {model_name}...")
    
    # Initialize the pipeline which will download and cache the model
    # We use CPU here just to force the download
    pipeline("automatic-speech-recognition", model=model_name, device="cpu")
    
    print(f"Successfully downloaded and cached {model_name}.")

if __name__ == "__main__":
    main()
