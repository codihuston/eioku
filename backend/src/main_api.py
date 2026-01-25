"""API Service entry point - FastAPI application without arq consumer."""

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.logger import logger as fastapi_logger

from src.api.artifact_controller import router as artifact_router
from src.api.path_controller_full import router as path_router
from src.api.task_routes import router as task_router
from src.api.video_controller import router as video_router
from src.database.connection import get_db
from src.database.migrations import run_migrations
from src.repositories.path_config_repository import SQLAlchemyPathConfigRepository
from src.services.config_loader import ConfigLoader
from src.services.job_producer import JobProducer
from src.services.path_config_manager import PathConfigManager
from src.services.video_discovery_service import VideoDiscoveryService
from src.utils.print_logger import get_logger

# Configure logging for gunicorn + uvicorn compatibility
gunicorn_error_logger = logging.getLogger("gunicorn.error")
gunicorn_logger = logging.getLogger("gunicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers

fastapi_logger.handlers = gunicorn_error_logger.handlers

if __name__ != "__main__":
    fastapi_logger.setLevel(gunicorn_logger.level)
else:
    fastapi_logger.setLevel(logging.DEBUG)

# Use print logger for startup
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events for API Service."""
    logger.info("🚀 API SERVICE STARTUP")

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

    try:
        logger.info("4️⃣ Loading config...")
        path_repo = SQLAlchemyPathConfigRepository(session)
        path_manager = PathConfigManager(path_repo)
        config_loader = ConfigLoader(path_manager)
        config_path = getattr(app.state, "config_path", None)
        config_loader.load_initial_config(config_path)
        logger.info("✅ Config loaded")

        logger.info("5️⃣ Importing repositories...")
        from src.repositories.task_repository import SQLAlchemyTaskRepository
        from src.repositories.video_repository import SqlVideoRepository

        logger.info("✅ Repositories imported")

        logger.info("6️⃣ Creating repositories...")
        video_repo = SqlVideoRepository(session)
        task_repo = SQLAlchemyTaskRepository(session)
        logger.info("✅ Repositories created")

        logger.info("7️⃣ Initializing job producer...")
        job_producer = JobProducer()
        await job_producer.initialize()
        logger.info("✅ Job producer initialized")

        logger.info("8️⃣ Running auto-discovery...")
        discovery_service = VideoDiscoveryService(
            path_manager, video_repo, job_producer
        )
        discovered_videos = discovery_service.discover_videos()
        logger.info(f"✅ Discovered {len(discovered_videos)} videos")

        logger.info("9️⃣ Auto-creating and enqueueing ML tasks for discovered videos...")
        tasks_created = 0
        for video in discovered_videos:
            if video.status == "discovered":
                try:
                    await discovery_service.discover_and_queue_tasks(video.file_path)
                    tasks_created += 6  # 6 tasks per video
                    logger.info(
                        f"✅ Auto-created and queued 6 ML tasks for "
                        f"video {video.video_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to auto-create tasks for video {video.video_id}: {e}",
                        exc_info=True,
                    )
                    # Continue with next video instead of failing entire startup

        await job_producer.close()
        logger.info(f"✅ Created and queued {tasks_created} ML tasks")

        logger.info("🔟 Loading pending tasks from database...")
        # Note: Pending tasks are now managed by arq job queue in Redis
        # The Worker Service will consume jobs from Redis and process them
        pending_tasks = task_repo.find_by_status("pending")
        logger.info(
            f"✅ Found {len(pending_tasks)} pending tasks in database "
            f"(will be processed by Worker Service via Redis)"
        )

        logger.info("1️⃣1️⃣ Storing in app state...")
        app.state.video_repo = video_repo
        app.state.task_repo = task_repo
        app.state.job_producer = job_producer
        logger.info("✅ API SERVICE STARTUP COMPLETE")

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise to prevent app from starting

    yield

    # Shutdown
    logger.info("🛑 API SERVICE SHUTTING DOWN...")
    if hasattr(app.state, "job_producer"):
        await app.state.job_producer.close()
    session.close()
    logger.info("✅ API SERVICE SHUTDOWN COMPLETE")


def create_app(config_path: str | None = None) -> FastAPI:
    """Create FastAPI application for API Service (no arq consumer)."""
    app = FastAPI(
        title="Eioku - Semantic Video Search API",
        description="API for semantic video search and processing",
        version="1.0.0",
        openapi_version="3.0.2",
        root_path="/api",  # Tell FastAPI about the reverse proxy prefix
        lifespan=lifespan,
    )

    # Store config path in app state
    if config_path:
        app.state.config_path = config_path

    # Include routers
    fastapi_logger.info("Including video router...")
    app.include_router(video_router, prefix="/v1")
    fastapi_logger.info("Including artifact router...")
    app.include_router(artifact_router, prefix="/v1")
    fastapi_logger.info("Including path router...")
    app.include_router(path_router, prefix="/v1")
    fastapi_logger.info("Including task router...")
    app.include_router(task_router, prefix="/v1")
    fastapi_logger.info("Routers included successfully")

    return app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Eioku API Service - Semantic Video Search Platform"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (default: /etc/eioku/config.json or "
        "EIOKU_CONFIG_PATH env var)",
    )
    return parser.parse_args()


# Create app instance
if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.config)
else:
    # For uvicorn/gunicorn - check sys.argv for config
    config_path = None
    if "--config" in sys.argv:
        try:
            config_idx = sys.argv.index("--config")
            if config_idx + 1 < len(sys.argv):
                config_path = sys.argv[config_idx + 1]
        except (ValueError, IndexError):
            pass

    app = create_app(config_path)


@app.get("/")
async def root():
    """Hello world endpoint."""
    return {"message": "Eioku API Service is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "api"}
