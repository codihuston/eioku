"""Health check endpoint."""

import logging

from fastapi import APIRouter

from ..models.responses import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(models_registry: dict = None) -> HealthResponse:
    """Get health status of ML Service.

    Returns:
        HealthResponse with service status and model information
    """
    import torch

    if models_registry is None:
        models_registry = {}

    # Check model status
    models_loaded = []
    models_failed = []

    for model_name, model_info in models_registry.items():
        if model_info.get("status") == "ready":
            models_loaded.append(model_name)
        else:
            models_failed.append(model_name)

    # Determine overall status
    if models_failed:
        status = "degraded" if models_loaded else "unhealthy"
    else:
        status = "healthy"

    # Get GPU info
    gpu_available = torch.cuda.is_available()
    gpu_device_name = None
    gpu_memory_total_mb = None
    gpu_memory_used_mb = None

    if gpu_available:
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_memory_total_mb = int(
            torch.cuda.get_device_properties(0).total_memory / 1e6
        )
        gpu_memory_used_mb = int(torch.cuda.memory_allocated(0) / 1e6)

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        gpu_available=gpu_available,
        gpu_device_name=gpu_device_name,
        gpu_memory_total_mb=gpu_memory_total_mb,
        gpu_memory_used_mb=gpu_memory_used_mb,
    )
