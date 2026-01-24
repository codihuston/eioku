"""Eioku ML Service - Stateless ML inference endpoints."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.health import router as health_router
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    global GPU_SEMAPHORE, INITIALIZATION_ERROR

    logger.info("🚀 ML Service starting up...")

    try:
        # Initialize GPU semaphore
        gpu_concurrency = int(os.getenv("GPU_CONCURRENCY", 2))
        GPU_SEMAPHORE = asyncio.Semaphore(gpu_concurrency)
        logger.info(f"GPU semaphore initialized with concurrency={gpu_concurrency}")

        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        require_gpu = os.getenv("REQUIRE_GPU", "false").lower() == "true"

        logger.info(f"GPU available: {gpu_available}")

        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU device: {device_name}")
            logger.info(f"GPU memory: {total_memory:.2f} GB")
        else:
            if require_gpu:
                raise RuntimeError(
                    "GPU required but not available (set REQUIRE_GPU=false to allow CPU-only mode)"
                )
            logger.warning(
                "GPU not available - will use CPU for inference (slower)"
            )

        # Initialize model manager
        model_cache_dir = os.getenv("MODEL_CACHE_DIR", "/models")
        manager = ModelManager(cache_dir=model_cache_dir)
        logger.info(f"Model cache directory: {model_cache_dir}")

        # Define models to initialize
        models_to_init = [
            ("yolov8n.pt", "yolo"),
            ("yolov8n-face.pt", "yolo"),
            ("large-v3", "whisper"),
            ("english", "easyocr"),
        ]

        # Initialize models
        for model_name, model_type in models_to_init:
            try:
                logger.info(f"Initializing {model_type} model: {model_name}")
                await manager.download_model(model_name, model_type)
                await manager.verify_model(model_name, model_type)
                MODELS_REGISTRY[model_name] = {
                    "status": "ready",
                    "type": model_type,
                }
                logger.info(f"✓ {model_name} initialized successfully")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {model_name}: {e}")
                MODELS_REGISTRY[model_name] = {
                    "status": "failed",
                    "type": model_type,
                    "error": str(e),
                }
                INITIALIZATION_ERROR = str(e)

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
app.include_router(health_router)


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
