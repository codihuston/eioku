# Eioku ML Service

Stateless ML inference endpoints for video analysis.

## Overview

The ML Service is a FastAPI-based microservice that provides HTTP endpoints for ML inference operations. It is designed to be:

- **Stateless**: No persistent state, can be scaled horizontally
- **Isolated**: No database access, only HTTP communication
- **GPU-aware**: Manages GPU resources via semaphore
- **Reproducible**: Includes provenance metadata (config_hash, input_hash)

## Architecture

```
┌─────────────────────────────────────────┐
│         ML Service (Port 8001)          │
├─────────────────────────────────────────┤
│  FastAPI Server                         │
│  - /infer/objects                       │
│  - /infer/faces                         │
│  - /infer/transcribe                    │
│  - /infer/ocr                           │
│  - /infer/places                        │
│  - /infer/scenes                        │
│  - /health                              │
├─────────────────────────────────────────┤
│  GPU Semaphore (concurrency control)    │
│  Model Cache (lazy-loaded)              │
└─────────────────────────────────────────┘
```

## Endpoints

### Health Check

```
GET /health
```

Returns service status and model information.

### Object Detection

```
POST /infer/objects
```

Detect objects in video using YOLO.

### Face Detection

```
POST /infer/faces
```

Detect faces in video using YOLO.

### Transcription

```
POST /infer/transcribe
```

Transcribe audio from video using Whisper.

### OCR

```
POST /infer/ocr
```

Extract text from video frames using EasyOCR.

### Place Detection

```
POST /infer/places
```

Classify places in video frames using Places365.

### Scene Detection

```
POST /infer/scenes
```

Detect scene boundaries in video.

## Environment Variables

- `GPU_CONCURRENCY`: Max concurrent GPU operations (default=2)
- `MODEL_CACHE_DIR`: Directory for model storage (default=/models)
- `REQUIRE_GPU`: Fail startup if GPU not available (default=false)
- `LOG_LEVEL`: Logging level (default=INFO)

## Development

### Setup

```bash
cd ml-service
poetry install
```

### Run

```bash
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### Test

```bash
poetry run pytest
```

## Docker

### Build

```bash
docker build -t eioku-ml-service:latest .
```

### Run

```bash
docker run -p 8001:8001 \
  -e GPU_CONCURRENCY=2 \
  -e MODEL_CACHE_DIR=/models \
  -v /models:/models \
  eioku-ml-service:latest
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU support)
