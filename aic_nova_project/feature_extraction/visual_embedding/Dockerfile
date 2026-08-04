FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY feature_extraction/visual_embedding/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY feature_extraction/visual_embedding /app/feature_extraction/visual_embedding
COPY scripts/download_pe_core.py /app/scripts/

# Set Python path so our module is importable
ENV PYTHONPATH=/app

# Download the model weights to cache for offline use
RUN python scripts/download_pe_core.py

# Default entrypoint to the CLI
ENTRYPOINT ["python", "-m", "feature_extraction.visual_embedding.cli"]
