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
├── retrieval_api/                     # Module 8: Online retrieval API and runtime wiring
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

## Testing

Each module contains its own self-contained test suite located in its `tests/` subdirectory.
To run all tests from the root:
```bash
# Add current directory to PYTHONPATH so packages can be resolved locally
$env:PYTHONPATH="."
pytest data_pipeline/shot_keyframe/tests feature_extraction/visual_embedding/tests feature_extraction/asr_transcript/tests feature_extraction/ocr/tests feature_extraction/object_detection/tests
```

### Online Data & Infrastructure

From PowerShell at the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r online\requirements-runtime.txt
python -m pip install -r online\requirements-test.txt
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

The contract/adapter unit tests do not require running Milvus or Elasticsearch.
Optional runtime validation is read-only:

```powershell
python -m online.validate_contract
python -m online.validate_contract --fail-on-partial
```

Configure `AIC_ONLINE_MILVUS_URI`, `AIC_ONLINE_ES_URI`, and
`AIC_ONLINE_SQLITE_PATH` (plus the other names documented in
`online/README.md`) instead of embedding endpoints or credentials in code.

* `feature_extraction/visual_embedding`: Generates feature vectors from keyframes.
* `feature_extraction/ocr`: Extracts and recognizes text overlay (subtitles, banners) in keyframes.
* `feature_extraction/asr_transcript`: Extracts audio, generates transcripts via PhoWhisper, and cleans transcripts with LLM.
* `feature_extraction/object_detection`: Detects objects using YOLO-World (open-vocabulary) and Co-DETR (COCO), with Box Fusion (NMS) support.
