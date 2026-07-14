import open_clip

if __name__ == "__main__":
    print("Downloading PE-Core-bigG-14-448 weights...")
    # This will download the weights and cache them in ~/.cache/huggingface/hub/
    open_clip.create_model_and_transforms("hf-hub:timm/PE-Core-bigG-14-448")
    print("Download complete.")
