"""arq worker configuration for Worker Service.

This module configures the arq worker to consume jobs from the 'jobs' queue.
The worker is stateless and doesn't care about GPU - it just:
1. Consumes jobs from Redis
2. Updates task status to RUNNING
3. Enqueues to ml_jobs for ML Service processing
4. Polls for results and persists artifacts
5. Runs periodic reconciliation to recover from failures

The ML Service handles all GPU concerns.
"""

import logging
import os

from arq import cron

from ..config.redis_config import REDIS_SETTINGS
from ..database.connection import get_db
from ..workers.reconciler import Reconciler

logger = logging.getLogger(__name__)


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
    - Periodic reconciliation task (every 5 minutes)
    """

    # Job handler functions (will be populated when handlers are implemented)
    functions = []

    # Worker consumes from the single 'ml_jobs' queue
    queue_name = "ml_jobs"

    # Redis connection settings (centralized in redis_config.py)
    redis_settings = REDIS_SETTINGS

    # Job processing configuration
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", 4))
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", 1800))  # 30 minutes
    max_tries = 3

    # Enable job abort for cancellation support
    allow_abort_jobs = True

    def __init__(self):
        """Initialize worker settings and log configuration."""
        logger.info(
            f"Worker configured: "
            f"queue={self.queue_name}, "
            f"max_jobs={self.max_jobs}, "
            f"job_timeout={self.job_timeout}s"
        )


# Periodic reconciliation task (every 5 minutes)
# Runs at minutes: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
cron_reconcile = cron(reconcile_tasks, minute=set(range(0, 60, 5)))
