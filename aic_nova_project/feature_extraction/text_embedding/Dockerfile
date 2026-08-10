FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY feature_extraction/text_embedding/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy download script
COPY scripts/download_text_models.py /app/scripts/

# Download the model weights to cache for offline use
ENV HF_HOME=/app/models
RUN python scripts/download_text_models.py --model-name dangvantuan/vietnamese-embedding --model-revision 4ab46e46ba5902328ba0742e489e75f787932f2b --cache-dir /app/models

# Copy source code
COPY feature_extraction/text_embedding/ /app/

# Set Python path
ENV PYTHONPATH=/app

# Default entrypoint to the CLI
ENTRYPOINT ["python", "-m", "src.text_embedding.cli"]
