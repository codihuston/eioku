"""Worker Service entry point - arq worker without HTTP endpoints."""

import asyncio
import logging
import os
from datetime import datetime, timedelta

from arq import cron
from arq.connections import RedisSettings

from src.config.redis_config import get_redis_settings
from src.database.connection import get_db
from src.database.migrations import run_migrations
from src.repositories.artifact_repository import SqlArtifactRepository
from src.repositories.task_repository import SQLAlchemyTaskRepository
from src.utils.print_logger import get_logger
from src.workers.reconciler import Reconciler
from src.workers.task_handler import process_ml_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = get_logger(__name__)


async def startup(ctx):
    """Initialize Worker Service on startup."""
    logger.info("🚀 WORKER SERVICE STARTUP")

    logger.info("1️⃣ Registering artifact schemas...")
    from src.domain.schema_initialization import register_all_schemas

    register_all_schemas()
    logger.info("✅ Artifact schemas registered")

    logger.info("2️⃣ Running migrations...")
    run_migrations()
    logger.info("✅ Migrations done")

    logger.info("3️⃣ Getting DB session...")
    session = next(get_db())
    logger.info("✅ DB session obtained")

    logger.info("4️⃣ Creating repositories...")
    task_repo = SQLAlchemyTaskRepository(session)
    artifact_repo = SqlArtifactRepository(session)
    logger.info("✅ Repositories created")

    logger.info("5️⃣ Initializing reconciler...")
    reconciler = Reconciler(task_repo, artifact_repo)
    logger.info("✅ Reconciler initialized")

    logger.info("6️⃣ Storing in context...")
    ctx["session"] = session
    ctx["task_repo"] = task_repo
    ctx["artifact_repo"] = artifact_repo
    ctx["reconciler"] = reconciler
    logger.info("✅ WORKER SERVICE STARTUP COMPLETE")


async def shutdown(ctx):
    """Clean up Worker Service on shutdown."""
    logger.info("🛑 WORKER SERVICE SHUTTING DOWN...")
    if "session" in ctx:
        ctx["session"].close()
    logger.info("✅ WORKER SERVICE SHUTDOWN COMPLETE")


async def reconciliation_task(ctx):
    """Periodic reconciliation task - runs every 5 minutes."""
    logger.info("🔄 Running reconciliation task...")
    try:
        reconciler = ctx.get("reconciler")
        if reconciler:
            await reconciler.reconcile()
            logger.info("✅ Reconciliation complete")
        else:
            logger.warning("⚠️ Reconciler not available in context")
    except Exception as e:
        logger.error(f"❌ Reconciliation failed: {e}", exc_info=True)


def get_queue_names() -> list[str]:
    """Determine which queues this worker should consume from based on GPU_MODE."""
    gpu_mode = os.getenv("GPU_MODE", "auto").lower()

    if gpu_mode == "gpu":
        logger.info("🎮 GPU mode enabled - consuming from gpu_jobs queue")
        return ["gpu_jobs"]
    elif gpu_mode == "cpu":
        logger.info("💻 CPU mode enabled - consuming from cpu_jobs queue")
        return ["cpu_jobs"]
    elif gpu_mode == "auto":
        # Auto-detect GPU
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("🎮 GPU detected - consuming from gpu_jobs queue")
                return ["gpu_jobs"]
            else:
                logger.info("💻 No GPU detected - consuming from cpu_jobs queue")
                return ["cpu_jobs"]
        except ImportError:
            logger.warning("⚠️ torch not available - defaulting to cpu_jobs queue")
            return ["cpu_jobs"]
    else:
        logger.warning(f"⚠️ Unknown GPU_MODE: {gpu_mode} - defaulting to cpu_jobs")
        return ["cpu_jobs"]


class WorkerSettings:
    """arq worker settings."""

    # Queue configuration
    queue_names = get_queue_names()

    # Job configuration
    max_jobs = int(os.getenv("WORKER_MAX_JOBS", "10"))
    job_timeout = int(os.getenv("WORKER_JOB_TIMEOUT", "3600"))  # 1 hour
    max_tries = int(os.getenv("WORKER_MAX_TRIES", "3"))

    # Redis configuration
    redis_settings = get_redis_settings()

    # Startup and shutdown
    on_startup = startup
    on_shutdown = shutdown

    # Cron tasks (periodic tasks)
    cron_jobs = [
        cron(reconciliation_task, minute=0, second=0),  # Every minute at :00
    ]

    # Logging
    log_level = logging.INFO

    # Worker identification
    worker_name = f"worker-{os.getenv('GPU_MODE', 'auto')}-{os.getenv('HOSTNAME', 'unknown')}"

    def __init__(self):
        """Initialize worker settings."""
        logger.info(f"Worker Settings:")
        logger.info(f"  - Queue names: {self.queue_names}")
        logger.info(f"  - Max jobs: {self.max_jobs}")
        logger.info(f"  - Job timeout: {self.job_timeout}s")
        logger.info(f"  - Max tries: {self.max_tries}")
        logger.info(f"  - Worker name: {self.worker_name}")


# Export for arq
functions = [process_ml_task, reconciliation_task]
