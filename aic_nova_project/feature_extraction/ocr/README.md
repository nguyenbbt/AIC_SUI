# OCR Module (Module 4)

This module is responsible for detecting and recognizing Vietnamese text from video keyframes for the AI Challenge 2026.

## Features
- **Two-stage Pipeline**: Uses EasyOCR (CRAFT) for robust text detection and VietOCR (`vgg_transformer`) for highly accurate Vietnamese text recognition.
- **Perspective Transform**: Corrects tilted bounding boxes using perspective transformations before feeding them to the recognizer, significantly improving accuracy.
- **Smart Text Ordering**: Uses spatial clustering to automatically sort scattered text boxes into natural reading order (top-to-bottom, left-to-right).
- **Fully Offline**: Designed to run purely offline inside a Docker container without relying on cloud APIs.
- **Resumable**: Safely skips already processed videos to support pausing and resuming large workloads.

## Setup & Running

This module is packaged as a Docker container. 

### Build

```bash
docker build -t ocr-module .
```
*(Note: Building the image will automatically download the required model weights into the container so it can run offline).*

### Run

Run the module by mounting your data directory and specifying the paths:

```bash
docker run --gpus all -v $(pwd)/data:/data ocr-module \
  --keyframe-dir /data/keyframes \
  --metadata-dir /data/metadata \
  --output-dir /data/ocr \
  --confidence-threshold 0.4 \
  --vietocr-backbone vgg_transformer
```

### CLI Arguments

- `--keyframe-dir`: Path to keyframes.
- `--metadata-dir`: Path to metadata JSONs.
- `--output-dir`: Path to save OCR results.
- `--width-ths`: (Default: 0.7) Merging threshold for EasyOCR detection.
- `--mag-ratio`: (Default: 1.5) Image magnification for EasyOCR.
- `--vietocr-backbone`: `vgg_transformer` (default) or `vgg_seq2seq`.
- `--confidence-threshold`: (Default: 0.4) Minimum score to keep recognized text.
- `--force`: Overwrite existing results.

## Output Format

The output is a JSON file for each video in the `--output-dir` with the following schema:
```json
{
  "video_id": "V001",
  "frames": [
    {
      "frame_id": "V001_00000_015",
      "shot_id": 0,
      "position": 0.15,
      "ocr_regions": [
        {
          "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
          "text": "Detected text",
          "confidence": 0.93
        }
      ],
      "ocr_text_concat": "Detected text"
    }
  ]
}
```
