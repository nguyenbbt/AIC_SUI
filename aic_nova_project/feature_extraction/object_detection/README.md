# Object Detection Module

This module is responsible for extracting object detection features from keyframes for the AI Challenge 2026 project. It supports running multiple advanced object detection models simultaneously and fuses their outputs using Non-Maximum Suppression (NMS).

## Features

- **Multi-Model Support**: Currently integrates YOLO-World (for open-vocabulary detection) and Co-DETR.
- **Box Fusion (NMS)**: Merges bounding boxes from multiple detectors using Intersection over Union (IoU) thresholds to prevent overlapping duplicate detections.
- **OOM Handling**: Automatic batch size reduction when CUDA Out Of Memory (OOM) errors occur.
- **Batch Processing**: Efficiently processes frames in batches to maximize GPU utilization.
- **Docker Support**: Ready-to-use Dockerfile for easy setup with all system dependencies and Python packages included.

## Prerequisites & Installation

### Option 1: Using Docker (Recommended)

The easiest way to run the module is via Docker, which automatically handles system dependencies like `libGL` and `ffmpeg`, and installs PyTorch with CUDA support.

```bash
# Build the Docker image
docker build -t object_detection -f feature_extraction/object_detection/Dockerfile .

# Run the Docker container
docker run --gpus all -v /path/to/data:/data object_detection \
    --keyframe-dir /data/keyframes \
    --metadata-dir /data/metadata \
    --output-dir /data/output \
    --run-yolo-world \
    --run-co-detr
```

### Option 2: Local Installation

1. Install Python dependencies:
```bash
pip install -r feature_extraction/object_detection/requirements.txt
```

2. Install `mmcv` and `mmdet` via `mim` (required for Co-DETR):
```bash
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.3.0"
```

3. Download model weights:
```bash
python scripts/download_object_detectors.py
```

## Usage

You can run the object detection pipeline through the provided CLI.

```bash
python -m src.object_detection.cli \
    --keyframe-dir /path/to/keyframes \
    --metadata-dir /path/to/metadata \
    --output-dir /path/to/output \
    --run-yolo-world \
    --yolo-world-model weights/yolov8s-world.pt \
    --run-co-detr \
    --co-detr-backbone resnet50 \
    --batch-size 16
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--keyframe-dir` | **(Required)** Directory containing input keyframes (saved in subdirectories per video). | |
| `--metadata-dir` | **(Required)** Directory containing metadata JSON files per video. | |
| `--output-dir` | **(Required)** Directory to save the extracted detection JSON files. | |
| `--run-yolo-world` | Flag to enable YOLO-World detector. | `False` |
| `--yolo-world-model` | Path to YOLO-World model weights. | `weights/yolov8s-world.pt` |
| `--custom-vocab-file`| Path to custom vocabulary TXT. If not set, defaults to COCO 80. | `None` |
| `--run-co-detr` | Flag to enable Co-DETR detector. | `False` |
| `--co-detr-backbone` | Backbone for Co-DETR (`resnet50` or `swin_l`). | `resnet50` |
| `--confidence-threshold` | Confidence threshold for detection filtering. | `0.25` |
| `--nms-threshold` | IoU threshold for Box Fusion (NMS). | `0.5` |
| `--batch-size` | Batch size for inference. | `16` |
| `--device` | Device to run inference on (`cuda` or `cpu`). | `cuda` |
| `--force` | Force re-processing even if the output file already exists. | `False` |

*Note: At least one detector (`--run-yolo-world` or `--run-co-detr`) must be enabled.*

## Input/Output Format

### Input
- **Keyframes**: `.webp` files structured as `<keyframe_dir>/<video_id>/<frame_id>.webp`.
- **Metadata**: JSON files containing a list of frames with their `frame_id`, `shot_id`, and `position`.

### Output
The module generates a JSON file for each processed video in the `output-dir` containing the detections:

```json
{
  "video_id": "V_001",
  "frames": [
    {
      "frame_id": "0001",
      "shot_id": "shot_0",
      "position": 1000,
      "objects": [
        {
          "label": "person",
          "score": 0.95,
          "box": [xmin, ymin, xmax, ymax],
          "area": 1500.5
        }
      ]
    }
  ]
}
```
