"""Eioku ML Service - Stateless ML inference endpoints."""

import logging
import logging.config
import os
from contextlib import asynccontextmanager

from pythonjsonlogger import jsonlogger


# A custom formatter to produce JSON logs
class JsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname.lower()
        log_record["name"] = record.name
        log_record["service"] = "ml-service"


def setup_logging():
    """
    Set up structured JSON logging for the entire application.
    This must be called before any other imports to ensure all loggers
    use JSON formatting from the start.
    """
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
        },
        "handlers": {
            "json_handler": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["json_handler"],
            "level": "INFO",
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["json_handler"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["json_handler"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(log_config)


# Set up logging immediately when the module is imported, BEFORE any other imports
setup_logging()

# Now import everything else that might use logging
import asyncio  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .api import health, inference  # noqa: E402
from .services.model_manager import ModelManager  # noqa: E402

# Get a logger instance for this module
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
                    "GPU required but not available "
                    "(set REQUIRE_GPU=false to allow CPU-only mode)"
                )
            logger.warning("GPU not available - will use CPU for inference (slower)")

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
                logger.info(
                    f"[{description}] Initializing {model_type} model: {model_name}"
                )
                await MODEL_MANAGER.download_model(model_name, model_type)
                await MODEL_MANAGER.verify_model(model_name, model_type)
                MODELS_REGISTRY[model_name] = {
                    "status": "ready",
                    "type": model_type,
                    "description": description,
                }
                logger.info(f"✓ [{description}] {model_name} initialized successfully")
            except Exception as e:
                logger.error(
                    f"✗ [{description}] Failed to initialize {model_name}: {e}"
                )
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
