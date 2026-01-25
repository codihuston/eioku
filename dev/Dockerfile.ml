# syntax=docker/dockerfile:1.4
# ML Service Dockerfile - runs inference models with GPU support

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Poetry
RUN python3 -m pip install --no-cache-dir poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_PATH=/opt/venv \
    POETRY_CACHE_DIR=/root/.cache/pypoetry

WORKDIR /app

# Copy Poetry files
COPY ml-service/pyproject.toml ml-service/poetry.lock* ./

# Install dependencies with BuildKit cache
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-root --only main

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY ml-service/src ./src

# Model cache directory
ENV MODEL_CACHE_DIR=/models \
    HF_HOME=/models/huggingface \
    YOLO_CONFIG_DIR=/models/ultralytics \
    EASYOCR_HOME=/models/easyocr

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Start the application
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
