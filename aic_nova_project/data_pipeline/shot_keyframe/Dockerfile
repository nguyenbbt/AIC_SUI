FROM python:3.10-slim

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# The supported build context is the project root.
COPY data_pipeline/shot_keyframe/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy only the package required by the service.
COPY data_pipeline/__init__.py /app/data_pipeline/__init__.py
COPY data_pipeline/shot_keyframe /app/data_pipeline/shot_keyframe

# Set Python path so `data_pipeline.shot_keyframe` package is importable
ENV PYTHONPATH=/app

# Set the entrypoint to the CLI
ENTRYPOINT ["python", "-m", "data_pipeline.shot_keyframe.cli"]
