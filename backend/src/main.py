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
    # Startup
    logger.info("FastAPI starting up...")
    run_migrations()

    # Load initial configuration
    session = next(get_db())
    try:
        path_repo = SQLAlchemyPathConfigRepository(session)
        path_manager = PathConfigManager(path_repo)
        config_loader = ConfigLoader(path_manager)

        # Use config path from app state if available
        config_path = getattr(app.state, "config_path", None)
        logger.info(f"Loading config from: {config_path}")
        config_loader.load_initial_config(config_path)

        # Start worker pool for task processing
        logger.info("Starting worker pool...")
        try:
            from src.repositories.task_repository import SQLAlchemyTaskRepository
            from src.repositories.video_repository import SqlVideoRepository
            from src.services.file_hash_service import FileHashService
            from src.services.task_orchestrator import TaskOrchestrator
            from src.services.worker_pool_manager import (
                HashWorker,
                WorkerPoolManager,
            )

            # Initialize repositories and orchestrator
            video_repo = SqlVideoRepository(session)
            task_repo = SQLAlchemyTaskRepository(session)
            orchestrator = TaskOrchestrator(task_repo, video_repo)

            # Create and start worker pool manager
            pool_manager = WorkerPoolManager(orchestrator)

            # Add hash worker pool
            hash_service = FileHashService()
            HashWorker(hash_service=hash_service)

            # Store in app state for access during runtime
            app.state.pool_manager = pool_manager
            app.state.orchestrator = orchestrator

            logger.info("Worker pool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to start worker pool: {e}")
            # Don't fail startup if worker pool fails

    finally:
        session.close()

    logger.info("FastAPI startup complete")
    yield
    # Shutdown
    logger.info("FastAPI shutting down")

    # Stop worker pools if they exist
    if hasattr(app.state, "pool_manager"):
        try:
            logger.info("Stopping worker pools...")
            # Add shutdown logic here when needed
        except Exception as e:
            logger.error(f"Error stopping worker pools: {e}")


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
