"""ML Service Worker - arq worker for processing ML jobs from Redis queue.

This service:
1. Consumes jobs from Redis ml_jobs queue
2. Processes ML inference tasks
3. Persists results to PostgreSQL
"""

import logging
import logging.config
import os

from pythonjsonlogger import jsonlogger


# A custom formatter to produce JSON logs
class JsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname.lower()
        log_record["name"] = record.name
        log_record["service"] = "ml-service-worker"


def setup_logging():
    """
    Set up structured JSON logging for the entire application.
    This function configures the root logger, and all other loggers will inherit
    this configuration. It also explicitly configures third-party loggers like
    Alembic, Uvicorn, and Gunicorn to use JSON formatting.
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
            "alembic": {
                "handlers": ["json_handler"],
                "level": "INFO",
                "propagate": False,
            },
            "alembic.runtime.migration": {
                "handlers": ["json_handler"],
                "level": "INFO",
                "propagate": False,
            },
            "arq": {
                "handlers": ["json_handler"],
                "level": "DEBUG",
                "propagate": False,
            },
            "arq.worker": {
                "handlers": ["json_handler"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(log_config)


# Set up logging immediately when the module is imported, BEFORE any other imports
setup_logging()

# Now import everything else that might use logging
from src.config.redis_config import REDIS_SETTINGS  # noqa: E402
from src.workers.task_handler import process_ml_task  # noqa: E402

logger = logging.getLogger(__name__)


async def startup(ctx):
    """Initialize ML Service Worker on startup."""
    logger.info("🚀 ML SERVICE WORKER STARTUP")

    logger.info("1️⃣ Registering artifact schemas...")
    from src.domain.schema_initialization import register_all_schemas

    register_all_schemas()
    logger.info("✅ Artifact schemas registered")

    logger.info("✅ ML SERVICE WORKER STARTUP COMPLETE")


async def shutdown(ctx):
    """Clean up ML Service Worker on shutdown."""
    logger.info("🛑 ML SERVICE WORKER SHUTTING DOWN...")
    logger.info("✅ ML SERVICE WORKER SHUTDOWN COMPLETE")


class App:
    """arq worker settings for ML Service.

    This worker:
    1. Consumes jobs from ml_jobs queue
    2. Runs ML inference using model manager
    3. Persists artifacts to PostgreSQL
    4. Does NOT handle reconciliation (backend worker does that)
    """

    # Queue configuration - worker consumes from ml_jobs queue
    queue_name = "ml_jobs"

    # Redis connection settings (centralized in redis_config.py)
    redis_settings = REDIS_SETTINGS

    # Job configuration
    max_jobs = int(os.getenv("WORKER_MAX_JOBS", "1"))
    job_timeout = int(os.getenv("WORKER_JOB_TIMEOUT", "3600"))  # 1 hour
    max_tries = int(os.getenv("WORKER_MAX_TRIES", "1"))  # No retries for now

    # Polling configuration
    poll_delay = 0.1  # Poll every 100ms

    # Shutdown
    on_startup = startup
    on_shutdown = shutdown

    # Functions to register with arq - ONLY job processing
    functions = [process_ml_task]

    # No cron tasks - reconciliation runs in backend worker
    cron_jobs = []

    # Logging
    log_level = logging.DEBUG

    # Worker identification
    worker_name = f"ml-worker-{os.getenv('HOSTNAME', 'unknown')}"


# Export for arq
App = App

# Export functions for arq to discover
functions = [process_ml_task]
