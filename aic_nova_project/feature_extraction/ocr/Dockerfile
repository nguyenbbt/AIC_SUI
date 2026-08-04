FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY feature_extraction/ocr/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create a small script to download models during build time
RUN python -c "import easyocr; reader = easyocr.Reader(['vi'], gpu=False, download_enabled=True); \
from vietocr.tool.predictor import Predictor; \
from vietocr.tool.config import Cfg; \
config = Cfg.load_config_from_name('vgg_transformer'); \
config['device'] = 'cpu'; \
Predictor(config); \
config_seq = Cfg.load_config_from_name('vgg_seq2seq'); \
config_seq['device'] = 'cpu'; \
Predictor(config_seq);"

# Copy application source code
COPY feature_extraction/ocr/src /app/src

# Set environment variables for Python
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

ENTRYPOINT ["python", "-m", "ocr_module.cli"]
