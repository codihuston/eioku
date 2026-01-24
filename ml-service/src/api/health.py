"""Health check endpoint."""

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter

from ..models.responses import HealthResponse

if TYPE_CHECKING:
    from ..services.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Global references (set by main.py)
MODELS_REGISTRY: dict = {}
MODEL_MANAGER: "ModelManager | None" = None


def set_globals(models_registry: dict, manager: "ModelManager"):
    """Set global references for health endpoint.

    Args:
        models_registry: Dictionary of model status
        manager: ModelManager instance
    """
    global MODELS_REGISTRY, MODEL_MANAGER
    MODELS_REGISTRY = models_registry
    MODEL_MANAGER = manager


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Get health status of ML Service.

    Returns:
        HealthResponse with service status and model information
    """
    # Check model status
    models_loaded = []
    models_failed = []

    for model_name, model_info in MODELS_REGISTRY.items():
        if model_info.get("status") == "ready":
            models_loaded.append(model_name)
        else:
            models_failed.append(model_name)

    # Determine overall status
    if models_failed:
        status = "degraded" if models_loaded else "unhealthy"
    else:
        status = "healthy"

    # Get GPU info from model manager
    gpu_info = {}
    if MODEL_MANAGER:
        gpu_info = MODEL_MANAGER.get_gpu_info()

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        gpu_available=gpu_info.get("gpu_available", False),
        gpu_device_name=gpu_info.get("gpu_device_name"),
        gpu_memory_total_mb=gpu_info.get("gpu_memory_total_mb"),
        gpu_memory_used_mb=gpu_info.get("gpu_memory_used_mb"),
    )
