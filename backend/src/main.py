import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.logger import logger as fastapi_logger

from src.api.path_controller_full import router as path_router
from src.api.task_routes import router as task_router
from src.api.video_controller import router as video_router
from src.database.connection import get_db
from src.database.migrations import run_migrations
from src.repositories.path_config_repository import SQLAlchemyPathConfigRepository
from src.services.config_loader import ConfigLoader
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
    """Handle application lifespan events."""
    logger.info("LIFESPAN STARTUP")

    logger.info("1️⃣ Running migrations...")
    run_migrations()
    logger.info("✅ Migrations done")

    logger.info("2️⃣ Getting DB session...")
    session = next(get_db())
    logger.info("✅ DB session obtained")

    try:
        logger.info("3️⃣ Loading config...")
        path_repo = SQLAlchemyPathConfigRepository(session)
        path_manager = PathConfigManager(path_repo)
        config_loader = ConfigLoader(path_manager)
        config_path = getattr(app.state, "config_path", None)
        config_loader.load_initial_config(config_path)
        logger.info("✅ Config loaded")

        logger.info("4️⃣ Importing services...")
        from src.repositories.task_repository import SQLAlchemyTaskRepository
        from src.repositories.video_repository import SqlVideoRepository
        from src.services.task_orchestration import TaskType
        from src.services.task_orchestrator import TaskOrchestrator
        from src.services.worker_pool_manager import (
            WorkerPoolManager,
        )

        logger.info("✅ Services imported")

        logger.info("5️⃣ Creating repositories...")
        video_repo = SqlVideoRepository(session)
        task_repo = SQLAlchemyTaskRepository(session)
        orchestrator = TaskOrchestrator(task_repo, video_repo)
        logger.info("✅ Repositories created")

        logger.info("6️⃣ Running auto-discovery...")
        discovery_service = VideoDiscoveryService(path_manager, video_repo)
        discovered_videos = discovery_service.discover_videos()
        logger.info(f"✅ Discovered {len(discovered_videos)} videos")

        logger.info("7️⃣ Creating tasks for discovered videos...")
        tasks_created = orchestrator.process_discovered_videos()
        logger.info(f"✅ Created {tasks_created} tasks for discovered videos")

        logger.info("8️⃣ Loading pending tasks from database...")
        # Load all pending tasks and enqueue them
        pending_tasks = task_repo.find_by_status("pending")
        for task in pending_tasks:
            task_type = TaskType(task.task_type)
            priority = orchestrator._get_task_priority(task_type)
            orchestrator.task_queues.enqueue(task, priority)
        logger.info(f"✅ Loaded {len(pending_tasks)} pending tasks into queues")

        logger.info("9️⃣ Loading processing profile...")
        from dataclasses import asdict

        from src.services.processing_profiles import ProfileManager

        profile_manager = ProfileManager()
        # Use balanced profile by default (can be made configurable later)
        profile = profile_manager.get_profile("balanced")
        task_settings_dict = asdict(profile.task_settings)
        logger.info(f"✅ Loaded profile: {profile.name}")
        logger.info(f"   - Model: {profile.task_settings.object_detection_model}")
        logger.info(
            f"   - Sample rate: {profile.task_settings.frame_sampling_interval}"
        )

        logger.info("🔟 Creating worker pool manager...")
        pool_manager = WorkerPoolManager(orchestrator, task_settings_dict)
        logger.info("✅ Pool manager created")

        logger.info("1️⃣1️⃣ Adding worker pools from profile...")
        for task_type_str, worker_config in profile.worker_configs.items():
            pool_manager.add_worker_pool(worker_config)
            logger.info(
                f"✅ Added {task_type_str} pool: {worker_config.worker_count} workers"
            )

        logger.info("1️⃣2️⃣ Starting all worker pools...")
        pool_manager.start_all()
        logger.info("✅ Worker pools started")

        logger.info("🏁 Storing in app state...")
        app.state.pool_manager = pool_manager
        app.state.orchestrator = orchestrator
        logger.info("✅ STARTUP COMPLETE")

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise to prevent app from starting

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    if hasattr(app.state, "pool_manager"):
        app.state.pool_manager.stop_all()
    session.close()
    logger.info("✅ SHUTDOWN COMPLETE")


def create_app(config_path: str | None = None) -> FastAPI:
    """Create FastAPI application with optional config path."""
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
    fastapi_logger.info("Including path router...")
    app.include_router(path_router, prefix="/v1")
    fastapi_logger.info("Including task router...")
    app.include_router(task_router, prefix="/v1")
    fastapi_logger.info("Routers included successfully")

    return app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Eioku Semantic Video Search Platform")
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
    return {"message": "Eioku API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
