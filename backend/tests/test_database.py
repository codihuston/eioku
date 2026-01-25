import os
import tempfile

from fastapi.testclient import TestClient

from src.database.migrations import run_migrations
from src.main_api import app


def test_migration_runner_creates_database():
    """Test that the migration runner creates database file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = os.path.join(temp_dir, "test.db")
        os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"

        try:
            # Run migrations directly
            run_migrations()

            # Verify the database file was created
            assert os.path.exists(test_db_path)

        finally:
            # Clean up environment
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]


def test_database_migrations_run_on_startup():
    """Test that database migrations run successfully on startup."""
    # Use a temporary database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = os.path.join(temp_dir, "test.db")
        os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"

        try:
            # Create test client with lifespan events enabled
            with TestClient(app) as client:
                # Verify the app responds
                response = client.get("/health")
                assert response.status_code == 200
                assert response.json() == {"status": "healthy"}

                # Verify the database file was created
                assert os.path.exists(test_db_path)

        finally:
            # Clean up environment
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]
