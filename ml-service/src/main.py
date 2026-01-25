"""Eioku ML Service - Stateless ML inference endpoints."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import health, inference
from .services.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
MODELS_REGISTRY = {}
GPU_SEMAPHORE = None
INITIALIZATION_ERROR = None
MODEL_MANAGER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    global GPU_SEMAPHORE, INITIALIZATION_ERROR, MODEL_MANAGER

    logger.info("🚀 ML Service starting up...")

    try:
        # Initialize GPU semaphore
        gpu_concurrency = int(os.getenv("GPU_CONCURRENCY", 2))
        GPU_SEMAPHORE = asyncio.Semaphore(gpu_concurrency)
        logger.info(f"GPU semaphore initialized with concurrency={gpu_concurrency}")

        # Initialize model manager
        model_cache_dir = os.getenv("MODEL_CACHE_DIR", "/models")
        MODEL_MANAGER = ModelManager(cache_dir=model_cache_dir)
        logger.info(f"Model cache directory: {model_cache_dir}")

        # Check GPU availability
        gpu_available = MODEL_MANAGER.detect_gpu()
        require_gpu = os.getenv("REQUIRE_GPU", "false").lower() == "true"

        # Set globals for inference and health endpoints
        inference.set_globals(GPU_SEMAPHORE, MODEL_MANAGER, gpu_available)
        health.set_globals(MODELS_REGISTRY, MODEL_MANAGER)

        logger.info(f"GPU available: {gpu_available}")

        if gpu_available:
            MODEL_MANAGER.log_gpu_info()
        else:
            if require_gpu:
                raise RuntimeError(
                    "GPU required but not available (set REQUIRE_GPU=false to allow CPU-only mode)"
                )
            logger.warning(
                "GPU not available - will use CPU for inference (slower)"
            )

        # Define models to initialize
        models_to_init = [
            ("yolov8n.pt", "yolo", "Object Detection"),
            ("yolov8n-face.pt", "yolo", "Face Detection"),
            ("large-v3", "whisper", "Transcription"),
            ("english", "easyocr", "OCR"),
        ]

        # Initialize models
        for model_name, model_type, description in models_to_init:
            try:
                logger.info(f"[{description}] Initializing {model_type} model: {model_name}")
                await MODEL_MANAGER.download_model(model_name, model_type)
                await MODEL_MANAGER.verify_model(model_name, model_type)
                MODELS_REGISTRY[model_name] = {
                    "status": "ready",
                    "type": model_type,
                    "description": description,
                }
                logger.info(f"✓ [{description}] {model_name} initialized successfully")
            except Exception as e:
                logger.error(f"✗ [{description}] Failed to initialize {model_name}: {e}")
                MODELS_REGISTRY[model_name] = {
                    "status": "failed",
                    "type": model_type,
                    "description": description,
                    "error": str(e),
                }
                INITIALIZATION_ERROR = str(e)

        # Check if any models failed
        failed_models = [
            m for m, info in MODELS_REGISTRY.items() if info["status"] == "failed"
        ]
        if failed_models:
            logger.warning(f"⚠️  {len(failed_models)} model(s) failed to initialize")
        else:
            logger.info("✅ All models initialized successfully")

        logger.info("✅ ML Service startup complete")

    except Exception as e:
        INITIALIZATION_ERROR = str(e)
        logger.error(f"❌ ML Service startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 ML Service shutting down...")
    logger.info("✅ ML Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Eioku ML Service",
    description="Stateless ML inference endpoints for video analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(inference.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Eioku ML Service",
        "version": "0.1.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
