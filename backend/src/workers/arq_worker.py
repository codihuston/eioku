"""arq worker configuration for Worker Service.

This module configures the arq worker to consume jobs from the single 'jobs'
queue. The worker reads the GPU_MODE environment variable to determine which
tasks it can handle:
- 'gpu': Only handle GPU-required tasks
- 'cpu': Only handle CPU-capable tasks
- 'auto': Auto-detect GPU and handle appropriate tasks

All jobs are enqueued to the single 'jobs' queue. Workers filter based on
their GPU_MODE capability.

The worker also runs a periodic reconciliation task every 5 minutes to:
- Detect and recover from Redis data loss (missing jobs for PENDING tasks)
- Detect and recover from job loss (RUNNING tasks with no job in Redis)
- Sync job completion status from Redis to PostgreSQL
- Alert on long-running tasks
"""

import logging
import os

import torch
from arq import cron

from ..config.redis_config import REDIS_SETTINGS
from ..database.connection import get_db
from ..workers.reconciler import Reconciler

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


async def reconcile_tasks(ctx) -> dict:
    """Periodic reconciliation task (runs every 5 minutes).

    This cron task synchronizes PostgreSQL task state with Redis job state:
    1. Checks all PENDING tasks and re-enqueues missing jobs (handles Redis data loss)
    2. Checks all RUNNING tasks and syncs with Redis state
    3. Alerts on long-running tasks (never auto-kills)

    Args:
        ctx: arq context with access to worker configuration

    Returns:
        Dictionary with reconciliation statistics
    """
    try:
        logger.info("Starting periodic reconciliation task")

        # Get database session
        session = next(get_db())

        try:
            # Create reconciler and run all checks
            reconciler = Reconciler(session=session)
            stats = await reconciler.run()

            logger.info(f"Reconciliation complete: {stats}")
            return stats

        finally:
            # Close database session
            session.close()

    except Exception as e:
        logger.error(f"Error in reconciliation task: {e}", exc_info=True)
        return {"error": str(e)}


class WorkerSettings:
    """arq worker configuration.

    This class configures the arq worker with:
    - Single 'jobs' queue for all tasks
    - Redis connection settings
    - Job processing parameters (max_jobs, timeout, retries)
    - Job abort capability for cancellation support
    - GPU_MODE to filter which tasks this worker can handle
    - Periodic reconciliation task (every 5 minutes)
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


# Periodic reconciliation task (every 5 minutes)
# Runs at minutes: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
cron_reconcile = cron(reconcile_tasks, minute=set(range(0, 60, 5)))
