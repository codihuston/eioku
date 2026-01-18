import argparse
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.video_controller import router as video_router
from src.database.connection import get_db
from src.database.migrations import run_migrations
from src.repositories.path_config_repository import SQLAlchemyPathConfigRepository
from src.services.config_loader import ConfigLoader
from src.services.path_config_manager import PathConfigManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    run_migrations()

    # Load initial configuration
    session = next(get_db())
    try:
        path_repo = SQLAlchemyPathConfigRepository(session)
        path_manager = PathConfigManager(path_repo)
        config_loader = ConfigLoader(path_manager)

        # Use config path from app state if available
        config_path = getattr(app.state, "config_path", None)
        config_loader.load_initial_config(config_path)
    finally:
        session.close()

    yield
    # Shutdown (nothing to do for now)


def create_app(config_path: str | None = None) -> FastAPI:
    """Create FastAPI application with optional config path."""
    app = FastAPI(
        title="Eioku - Semantic Video Search API",
        description="API for semantic video search and processing",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store config path in app state
    if config_path:
        app.state.config_path = config_path

    # Include routers
    app.include_router(video_router, prefix="/v1")

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
