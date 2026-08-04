# AI Challenge 2026 - Nova Project (Video Retrieval System)

This repository contains the full pipeline for the AI Challenge 2026 Video Retrieval System. The system is architected as a set of independent microservices/modules, each containerized via Docker.

## Directory Architecture

```text
<project-root>/
├── logs/                              # Contains all execution logs (e.g., preprocessing.log)
├── weights/                           # Model weights cache for offline operation
├── scripts/                           # Shared utility scripts (downloads, etc.)
├── notebooks/                         # Experimental notebooks
├── data/                              # Data directory (ignored by git)
│   ├── raw_videos/
│   └── processed/
│       ├── metadata/
│       ├── keyframes/
│       ├── audio/
│       ├── transcripts/
│       ├── summaries/
│       ├── ocr/
│       ├── embeddings/
│       ├── objects/
│       └── metadata_index.parquet
│
├── data_pipeline/
│   └── shot_keyframe/                 # Module 1: Shot detection & keyframe extraction
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── src/
│       └── tests/
│
├── feature_extraction/
│   ├── visual_embedding/              # Module 2: Visual feature embedding extraction
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   └── tests/
│   ├── asr_transcript/                # Module 3: Audio extraction, ASR, and summarization
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   └── tests/
│   ├── ocr/                           # Module 4: On-screen text extraction
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   └── tests/
│   └── object_detection/              # Module 5: Object detection (YOLO-World, Co-DETR)
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── src/
│       └── tests/
│
├── indexing/                          # Module 6: Vector DB and metadata indexing (scaffolded)
├── query_understanding/               # Module 7: Query expansion & intent (scaffolded)
├── retrieval_api/                     # Module 8: Core retrieval engine (scaffolded)
└── ui/                                # Module 9: Frontend User Interface (scaffolded)
```

## Running the Pipeline

Each module is self-contained. To build a specific module's Docker image, you must run the build command from the **project root** to ensure shared directories (like `scripts/`) are available in the context.

Example for Visual Embedding:
```bash
docker build -f feature_extraction/visual_embedding/Dockerfile -t visual_embedding .
```

Example for Data Pipeline:
```bash
docker build -f data_pipeline/shot_keyframe/Dockerfile -t shot_keyframe .
```

### Running Offline modules on Modal

`scripts/offline_modal_runner.py` exposes the real CLI entrypoints for Modules 1-7. Inputs and
outputs live in the `aic-nova-offline-data` Volume mounted at `/data`:

```bash
modal run scripts/offline_modal_runner.py --module module1 --arguments="--input /data/raw_videos --output /data/processed"
```

Select `module2` through `module7` and pass that module's normal CLI arguments
in `--arguments`. The runner does not start an HTTP service; Offline modules are
batch jobs.

For the checked Docker Desktop, VS Code extension, Modal Volume, and M-GPUX
Sandbox workflow, see [docs/MGPUX_DOCKER_GUIDE.md](docs/MGPUX_DOCKER_GUIDE.md).

## Testing

Each module contains its own self-contained test suite located in its `tests/`
subdirectory. To run every Offline module suite plus root cross-module and
Online contract tests from the repository root:
```bash
python scripts/run_all_tests.py -q
```

The runner discovers every repository `tests/` directory and executes each in
an isolated pytest process. This prevents collisions between modules that use
independent `src/` package layouts. Use `python scripts/run_all_tests.py --list`
to inspect the discovered suites without running them.

* `feature_extraction/visual_embedding`: Generates feature vectors from keyframes.
* `feature_extraction/ocr`: Extracts and recognizes text overlay (subtitles, banners) in keyframes.
* `feature_extraction/asr_transcript`: Extracts audio, generates transcripts via PhoWhisper, and cleans transcripts with LLM.
* `feature_extraction/object_detection`: Detects objects using YOLO-World (open-vocabulary) and Co-DETR (COCO), with Box Fusion (NMS) support.
