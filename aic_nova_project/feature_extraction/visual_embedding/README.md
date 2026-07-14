# Module 2: Visual Embedding

This module extracts visual embeddings from video keyframes using the `PE-Core-bigG-14-448` model from Meta (via `open_clip`).

## Features
- Complete offline capability (model weights are downloaded during Docker build).
- Fast multi-worker data loading.
- Efficient Parquet output format containing both metadata and vector embeddings.
- Automatic OOM handling (reduces batch size and retries on OOM errors).
- Graceful error handling for missing/corrupt images.
- Resume capability to skip already-processed videos.

## Requirements
To run locally outside of Docker:
```bash
pip install -r requirements.txt
```

## Running Locally
```powershell
python -m feature_extraction.visual_embedding.cli --metadata-dir data/metadata --keyframe-dir data/keyframes --output-dir data/embeddings --device auto --batch-size 64 --num-workers 4
```
```
python -m feature_extraction.visual_embedding.cli --metadata-dir data/metadata --keyframe-dir data/keyframes --output-dir data/embeddings --device cuda --model-id hf-hub:timm/ViT-L-14-SigLIP --batch-size 16 --num-workers 
```

## Running via Docker (Offline execution)
The Dockerfile is designed to cache the model weights during the build process, so no internet connection is required when running the container.

### Build
```bash
docker build -t visual-embedding .
```

### Run (with GPU)
```bash
docker run --gpus all -v /absolute/path/to/data:/data visual-embedding \
    --metadata-dir /data/metadata \
    --keyframe-dir /data/keyframes \
    --output-dir /data/embeddings \
    --device cuda \
    --batch-size 64
```

### Run (without GPU - CPU fallback)
```bash
docker run -v /absolute/path/to/data:/data visual-embedding \
    --metadata-dir /data/metadata \
    --keyframe-dir /data/keyframes \
    --output-dir /data/embeddings \
    --device cpu \
    --batch-size 16
```

## Output Format
The module outputs a Parquet file for each video (e.g., `V001.parquet`).
The schema of the Parquet file includes:
- `frame_id` (str): Unique identifier for the keyframe (e.g. `V001_00000_015`).
- `video_id` (str): Video ID.
- `shot_id` (int): Shot ID.
- `position` (float): Position inside the shot (percentage/ratio).
- `file_path` (str): Absolute file path to the extracted image.
- `model_name` (str): Name of the embedding model used.
- `embedding_dim` (int): Dimension of the embedding vector.
- `embedding` (list[float32]): L2-normalized visual embedding vector.
