"""arq worker configuration for Worker Service.

This module configures the arq worker to consume jobs from the single 'jobs'
queue. The worker reads the GPU_MODE environment variable to determine which
tasks it can handle:
- 'gpu': Only handle GPU-required tasks
- 'cpu': Only handle CPU-capable tasks
- 'auto': Auto-detect GPU and handle appropriate tasks

All jobs are enqueued to the single 'jobs' queue. Workers filter based on
their GPU_MODE capability.
"""

import logging
import os

import torch

from ..config.redis_config import REDIS_SETTINGS

logger = logging.getLogger(__name__)


# Determine GPU mode from environment
GPU_MODE = os.getenv("GPU_MODE", "auto")  # gpu, cpu, or auto


def get_gpu_mode() -> str:
    """Determine the GPU mode for this worker.

    Returns:
        GPU mode: 'gpu', 'cpu', or 'auto'

    Raises:
        ValueError: If GPU_MODE is not one of 'gpu', 'cpu', or 'auto'
    """
    if GPU_MODE not in ("gpu", "cpu", "auto"):
        raise ValueError(
            f"Invalid GPU_MODE: {GPU_MODE}. Must be 'gpu', 'cpu', or 'auto'"
        )

    if GPU_MODE == "auto":
        # Auto-detect GPU availability
        if torch.cuda.is_available():
            logger.info(
                f"GPU_MODE=auto: GPU detected ({torch.cuda.get_device_name(0)}) - "
                "will handle GPU-required tasks"
            )
            return "gpu"
        else:
            logger.info(
                "GPU_MODE=auto: GPU not available - will handle CPU-capable tasks"
            )
            return "cpu"

    return GPU_MODE


class WorkerSettings:
    """arq worker configuration.

    This class configures the arq worker with:
    - Single 'jobs' queue for all tasks
    - Redis connection settings
    - Job processing parameters (max_jobs, timeout, retries)
    - Job abort capability for cancellation support
    - GPU_MODE to filter which tasks this worker can handle
    """

    # Job handler functions (will be populated when handlers are implemented)
    functions = []

    # All workers consume from the single 'jobs' queue
    # They filter based on GPU_MODE capability
    queue_name = "jobs"

    # Redis connection settings (centralized in redis_config.py)
    redis_settings = REDIS_SETTINGS

    # Job processing configuration
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", 4))
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", 1800))  # 30 minutes
    max_tries = 3

    # Enable job abort for cancellation support
    allow_abort_jobs = True

    # GPU mode for task filtering
    gpu_mode = get_gpu_mode()

    def __init__(self):
        """Initialize worker settings and log configuration."""
        logger.info(
            f"Worker configured: GPU_MODE={GPU_MODE}, "
            f"effective_mode={self.gpu_mode}, "
            f"queue={self.queue_name}, "
            f"max_jobs={self.max_jobs}, "
            f"job_timeout={self.job_timeout}s"
        )
