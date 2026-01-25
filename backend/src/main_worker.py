"""Worker Service entry point - arq worker without HTTP endpoints."""

import logging
import os

from arq import cron

from src.config.redis_config import get_redis_settings
from src.database.connection import get_db
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

    logger.info("2️⃣ Getting DB session...")
    session = next(get_db())
    logger.info("✅ DB session obtained")

    logger.info("3️⃣ Creating repositories...")
    from src.domain.schema_registry import SchemaRegistry
    from src.services.projection_sync_service import ProjectionSyncService

    schema_registry = SchemaRegistry()
    projection_sync = ProjectionSyncService(session)
    task_repo = SQLAlchemyTaskRepository(session)
    artifact_repo = SqlArtifactRepository(session, schema_registry, projection_sync)
    logger.info("✅ Repositories created")

    logger.info("4️⃣ Initializing reconciler...")
    reconciler = Reconciler(session)
    logger.info("✅ Reconciler initialized")

    logger.info("5️⃣ Storing in context...")
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


class App:
    """arq worker settings."""

    # Queue configuration - worker consumes from jobs queue
    queue_names = ["jobs"]

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
    worker_name = f"worker-{os.getenv('HOSTNAME', 'unknown')}"

    def __init__(self):
        """Initialize worker settings."""
        logger.info("Worker Settings:")
        logger.info(f"  - Queue names: {self.queue_names}")
        logger.info(f"  - Max jobs: {self.max_jobs}")
        logger.info(f"  - Job timeout: {self.job_timeout}s")
        logger.info(f"  - Max tries: {self.max_tries}")
        logger.info(f"  - Worker name: {self.worker_name}")


# Export for arq
App = App

# Export functions for arq
functions = [process_ml_task, reconciliation_task]
