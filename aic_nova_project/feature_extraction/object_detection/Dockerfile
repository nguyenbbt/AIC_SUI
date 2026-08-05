FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY feature_extraction/object_detection/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install mmcv and mmdet via mim
RUN mim install "mmcv>=2.0.0" && \
    mim install "mmdet>=3.3.0"

# Copy source code and scripts
COPY feature_extraction/object_detection/ /app/
COPY scripts/download_object_detectors.py /app/scripts/

# Set Python path
ENV PYTHONPATH=/app

# Download the model weights to cache for offline use
RUN python scripts/download_object_detectors.py

# Default entrypoint to the CLI
ENTRYPOINT ["python", "-m", "src.object_detection.cli"]
