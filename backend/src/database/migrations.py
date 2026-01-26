import logging
import os

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)


def run_migrations():
    """Run database migrations on startup."""
    logger.info("Starting database migrations...")

    try:
        # Get database URL from environment or use default
        database_url = os.getenv("DATABASE_URL", "sqlite:///./data/eioku.db")

        # Ensure data directory exists for SQLite only
        if database_url.startswith("sqlite"):
            db_path = database_url.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir and db_dir != ".":
                os.makedirs(db_dir, exist_ok=True)

        # Get alembic config
        alembic_cfg = Config("alembic.ini")

        # Override the database URL in alembic config
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        # Run migrations
        db_type = "PostgreSQL" if database_url.startswith("postgresql") else "SQLite"
        logger.info(f"Running alembic migrations ({db_type})...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic upgrade completed")

        logger.info("Database migrations completed successfully")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
