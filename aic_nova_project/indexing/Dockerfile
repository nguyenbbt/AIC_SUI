FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements
COPY indexing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY indexing/ /app/
COPY verify_frame_id_consistency.py /app/

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.indexing.cli"]
