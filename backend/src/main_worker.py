"""Worker Service entry point - arq worker without HTTP endpoints."""

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
        log_record["service"] = "worker"


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
from src.workers.arq_worker import (  # noqa: E402
    WorkerSettings,
    cron_reconcile,
    reconcile_tasks,
)

logger = logging.getLogger(__name__)


async def startup(ctx):
    """Initialize Worker Service on startup."""
    logger.info("🚀 BACKEND WORKER SERVICE STARTUP")

    logger.info("1️⃣ Registering artifact schemas...")
    from src.domain.schema_initialization import register_all_schemas

    register_all_schemas()
    logger.info("✅ Artifact schemas registered")

    logger.info("2️⃣ Initializing reconciler...")
    from src.database.connection import SessionLocal
    from src.workers.reconciler import Reconciler

    session = SessionLocal()
    reconciler = Reconciler(session)
    logger.info("✅ Reconciler initialized")

    logger.info("3️⃣ Storing in context...")
    ctx["reconciler"] = reconciler
    ctx["reconciler_session"] = session
    logger.info("✅ BACKEND WORKER SERVICE STARTUP COMPLETE")


async def shutdown(ctx):
    """Clean up Worker Service on shutdown."""
    logger.info("🛑 BACKEND WORKER SERVICE SHUTTING DOWN...")
    if "reconciler_session" in ctx:
        ctx["reconciler_session"].close()
    logger.info("✅ BACKEND WORKER SERVICE SHUTDOWN COMPLETE")


class App(WorkerSettings):
    """arq worker settings for backend worker.

    This worker:
    1. Consumes from ml_jobs queue (same queue as ml-service)
    2. Runs periodic reconciliation to recover from failures
    3. Does NOT process jobs - ml-service handles that
    """

    # Shutdown handlers
    on_startup = startup
    on_shutdown = shutdown

    # Functions to register with arq
    functions = [reconcile_tasks]

    # Cron tasks (periodic reconciliation every 5 minutes)
    cron_jobs = [cron_reconcile]

    # Logging
    log_level = logging.DEBUG

    # Worker identification
    worker_name = f"backend-worker-{os.getenv('HOSTNAME', 'unknown')}"


# Export for arq
App = App

# Export functions for arq to discover
functions = [reconcile_tasks]
