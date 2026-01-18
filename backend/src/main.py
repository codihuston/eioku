import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.logger import logger

from src.api.path_controller_full import router as path_router
from src.api.task_routes import router as task_router
from src.api.video_controller import router as video_router
from src.database.connection import get_db
from src.database.migrations import run_migrations
from src.repositories.path_config_repository import SQLAlchemyPathConfigRepository
from src.services.config_loader import ConfigLoader
from src.services.path_config_manager import PathConfigManager
from src.services.video_discovery_service import VideoDiscoveryService

# Configure logging for gunicorn + uvicorn compatibility
gunicorn_error_logger = logging.getLogger("gunicorn.error")
gunicorn_logger = logging.getLogger("gunicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers

logger.handlers = gunicorn_error_logger.handlers

if __name__ != "__main__":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    print("🚀 LIFESPAN STARTUP", flush=True)

    print("1️⃣ Running migrations...", flush=True)
    run_migrations()
    print("✅ Migrations done", flush=True)

    print("2️⃣ Getting DB session...", flush=True)
    session = next(get_db())
    print("✅ DB session obtained", flush=True)

    try:
        print("3️⃣ Loading config...", flush=True)
        path_repo = SQLAlchemyPathConfigRepository(session)
        path_manager = PathConfigManager(path_repo)
        config_loader = ConfigLoader(path_manager)
        config_path = getattr(app.state, "config_path", None)
        config_loader.load_initial_config(config_path)
        print("✅ Config loaded", flush=True)

        print("4️⃣ Importing services...", flush=True)
        from src.repositories.task_repository import SQLAlchemyTaskRepository
        from src.repositories.video_repository import SqlVideoRepository
        from src.services.task_orchestration import TaskType
        from src.services.task_orchestrator import TaskOrchestrator
        from src.services.worker_pool_manager import (
            ResourceType,
            WorkerConfig,
            WorkerPoolManager,
        )

        print("✅ Services imported", flush=True)

        print("5️⃣ Creating repositories...", flush=True)
        video_repo = SqlVideoRepository(session)
        task_repo = SQLAlchemyTaskRepository(session)
        orchestrator = TaskOrchestrator(task_repo, video_repo)
        print("✅ Repositories created", flush=True)

        print("6️⃣ Running auto-discovery...", flush=True)
        discovery_service = VideoDiscoveryService(path_manager, video_repo)
        discovered_videos = discovery_service.discover_videos()
        print(f"✅ Discovered {len(discovered_videos)} videos", flush=True)

        print("7️⃣ Creating tasks for discovered videos...", flush=True)
        tasks_created = orchestrator.process_discovered_videos()
        print(f"✅ Created {tasks_created} tasks for discovered videos", flush=True)

        print("8️⃣ Creating worker pool manager...", flush=True)
        pool_manager = WorkerPoolManager(orchestrator)
        print("✅ Pool manager created", flush=True)

        print("9️⃣ Adding hash worker pool...", flush=True)
        hash_config = WorkerConfig(TaskType.HASH, 2, ResourceType.CPU, 1)
        pool_manager.add_worker_pool(hash_config)
        print("✅ Hash pool added", flush=True)

        print("🔟 Adding transcription worker pool...", flush=True)
        transcription_config = WorkerConfig(
            TaskType.TRANSCRIPTION, 1, ResourceType.CPU, 1
        )
        pool_manager.add_worker_pool(transcription_config)
        print("✅ Transcription pool added", flush=True)

        print("1️⃣1️⃣ Starting all worker pools...", flush=True)
        pool_manager.start_all()
        print("✅ Worker pools started", flush=True)

        print("🏁 Storing in app state...", flush=True)
        app.state.pool_manager = pool_manager
        app.state.orchestrator = orchestrator
        print("✅ STARTUP COMPLETE", flush=True)

    except Exception as e:
        print(f"❌ Error at step: {e}", flush=True)
        import traceback

        traceback.print_exc()

    finally:
        session.close()

    yield

    # Shutdown
    if hasattr(app.state, "pool_manager"):
        app.state.pool_manager.stop_all()
    print("✅ SHUTDOWN COMPLETE", flush=True)


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
    logger.info("Including video router...")
    app.include_router(video_router, prefix="/v1")
    logger.info("Including path router...")
    app.include_router(path_router, prefix="/v1")
    logger.info("Including task router...")
    app.include_router(task_router, prefix="/v1")
    logger.info("Routers included successfully")

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
