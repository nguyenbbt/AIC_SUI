FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY feature_extraction/asr_transcript/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY feature_extraction/asr_transcript/ /app/
COPY scripts/download_phowhisper.py /app/scripts/

# Set Python path so our module is importable
ENV PYTHONPATH=/app

# Download the model weights to cache for offline use
RUN python scripts/download_phowhisper.py --size medium

# Default entrypoint to the CLI
ENTRYPOINT ["python", "-m", "cli"]
