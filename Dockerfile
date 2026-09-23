# Use official lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install curl for container healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-optimized PyTorch and dependencies (with timeout & retry safety)
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

# Copy source code, trained models, and static UI
COPY src/ /app/src/
COPY models/ /app/models/
COPY static/ /app/static/
COPY data/processed/ /app/data/processed/

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI production server
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
