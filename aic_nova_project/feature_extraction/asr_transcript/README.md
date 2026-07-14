# Module 3: ASR & Transcript Cleaning/Summarization

This module extracts audio from raw videos, performs Automatic Speech Recognition (ASR) to generate timestamps and text, cleans the noisy text using Large Language Models (LLM), and creates video-level summaries.

## Features
- **Offline Capability**: Can run completely offline using local models (e.g., Qwen) for both transcription and cleaning/summarization.
- **Cloud LLM Support**: Defaults to Google Gemini (`gemini-2.5-flash`) via the `google-genai` SDK for ultra-fast, cheap, and accurate text cleaning. Uses Structured Outputs (JSON Schema) to guarantee valid responses.
- **Resumable**: Skips processing if output files already exist.
- **Automatic Fallbacks**: Parses existing `.srt`/`.vtt` captions if provided; otherwise falls back to PhoWhisper for Vietnamese ASR.
- **Concurrency**: Processes LLM cleaning API calls asynchronously to significantly reduce wait times.

## Requirements

Ensure you have installed the global `requirements.txt` dependencies.
For this module specifically, you need:
- `transformers`
- `google-genai`
- `tenacity`
- `ffmpeg` (Must be installed on the system)

### Setting up the API Key
**If using Gemini (default)**, set `GEMINI_API_KEY`:
- **Windows**: `$env:GEMINI_API_KEY="your_api_key_here"`
- **Linux/Mac**: `export GEMINI_API_KEY="your_api_key_here"`

**If using Azure OpenAI**, set the following:
- **Windows**:
  ```powershell
  $env:OPENAI_API_KEY="your_api_key_here"
  $env:BASE_URL="https://your-resource.openai.azure.com"
  $env:API_VERSION="2024-08-01-preview"
  ```
- **Linux/Mac**:
  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  export BASE_URL="https://your-resource.openai.azure.com"
  export API_VERSION="2024-08-01-preview"
  ```

## Usage

### 1. Run via Docker (Recommended)
Build the image:
```bash
docker build -t aic-pipeline .
```
Run the container:
```bash
docker run --gpus all -v /absolute/path/to/data:/data -e GEMINI_API_KEY=$GEMINI_API_KEY aic-pipeline \
    -m feature_extraction.asr_transcript.cli \
    --video-dir /data/raw_videos \
    --metadata-dir /data/metadata \
    --caption-dir /data/captions \
    --output-dir /data \
    --whisper-size medium \
    --llm-provider gemini
```

### 2. Run Locally
```powershell
python -m feature_extraction.asr_transcript.cli `
    --video-dir data/raw_videos `
    --metadata-dir data/processed/metadata `
    --caption-dir data/captions `
    --output-dir data/processed `
    --whisper-size medium `
    --llm-provider gemini

python -m feature_extraction.asr_transcript.cli `
    --video-dir data/raw_videos `
    --metadata-dir data/processed/metadata `
    --caption-dir data/captions `
    --output-dir data/processed `
    --whisper-size medium `
    --llm-provider azure `
    --llm-model azure/gpt-4o `
    --device cuda 

```

## Options
- `--llm-provider`: Choose `gemini` (default), `azure`, or `local` (offline).
- `--whisper-size`: `tiny`, `small`, `medium` (default), `large`.
- `--group-size`: Number of ASR segments to group into a single cleaning interval (default: 5).
- `--concurrency`: Number of parallel LLM calls (default: 10).
- `--device`: `auto` (default - uses GPU if available), `cpu`, or `cuda`. Controls where the local ASR model (and local LLM) runs.

## Outputs
- `data/audio/{video_id}.wav`: Extracted 16kHz mono audio.
- `data/transcripts/{video_id}_raw.json`: Raw segments from ASR/Captions.
- `data/transcripts/{video_id}_cleaned.json`: Grouped and LLM-cleaned text.
- `data/summaries/{video_id}.json`: Video-level summary.
