FROM python:3.10-slim

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Copy weights (assuming they are downloaded locally before docker build)
# If the directory doesn't exist, it won't fail if we just copy the whole weights dir
COPY weights/ /app/weights/

# Set Python path so `shot_keyframe` package is importable
ENV PYTHONPATH=/app/src

# Set the entrypoint to the CLI
ENTRYPOINT ["python", "-m", "shot_keyframe.cli"]
