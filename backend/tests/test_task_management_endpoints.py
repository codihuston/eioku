"""Tests for task management endpoints (cancel, retry, list).

Tests for:
- POST /tasks/{task_id}/cancel - Cancel a task
- POST /tasks/{task_id}/retry - Retry a failed/cancelled task
- GET /tasks - List tasks with filtering and sorting
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.main_api import app
from src.repositories.task_repository import SQLAlchemyTaskRepository

# Set testing mode to prevent worker pools from starting
os.environ["TESTING"] = "true"


@pytest.fixture(scope="module")
def test_db():
    """Create a temporary database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield testing_session_local

    os.unlink(db_path)


@pytest.fixture(scope="module")
def client(test_db):
    """Create test client with database dependency override."""

    def override_get_db():
        db = test_db()
        try:
            yield db
        finally:
            db.close()

    from src.database.connection import get_db

    app.dependency_overrides[get_db] = override_get_db

    # Mock Redis/arq connections to avoid connection errors during tests
    with patch("src.services.job_producer.create_pool") as mock_create_pool:
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def db_session(test_db):
    """Create a fresh database session for each test."""
    session = test_db()
    yield session
    session.close()


@pytest.fixture
def sample_video(db_session):
    """Create a sample video for testing."""
    from datetime import datetime

    from src.database.models import Video

    video = Video(
        video_id="test-video-1",
        filename="test.mp4",
        file_path="/test/test.mp4",
        file_hash="abc123",
        status="hashed",
        duration=120.0,
        file_size=1024000,
        last_modified=datetime.utcnow(),
    )
    db_session.add(video)
    db_session.commit()
    return video


@pytest.fixture
def sample_task(db_session, sample_video):
    """Create a sample task for testing."""
    from src.database.models import Task

    task = Task(
        task_id="test-task-1",
        video_id=sample_video.video_id,
        task_type="object_detection",
        status="pending",
        priority=1,
        dependencies=[],
    )
    db_session.add(task)
    db_session.commit()
    return task


class TestCancelTask:
    """Tests for POST /tasks/{task_id}/cancel endpoint."""

    def test_cancel_pending_task(self, client, db_session, sample_task):
        """Test cancelling a task in PENDING status."""
        task_id = sample_task.task_id

        with patch("src.api.task_routes.create_pool") as mock_create_pool:
            # Mock Redis pool
            mock_pool = AsyncMock()
            mock_job = AsyncMock()
            mock_pool.job.return_value = mock_job
            mock_create_pool.return_value = mock_pool

            response = client.post(f"/tasks/{task_id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "cancelled"
        assert "already running" in data["message"].lower()

        # Verify task status updated in database
        task_repo = SQLAlchemyTaskRepository(db_session)
        updated_task = task_repo.find_by_id(task_id)
        assert updated_task.status == "cancelled"

    def test_cancel_running_task(self, client, db_session, sample_task):
        """Test cancelling a task in RUNNING status."""
        # Update task to RUNNING
        sample_task.status = "running"
        sample_task.started_at = datetime.utcnow()
        db_session.commit()

        task_id = sample_task.task_id

        with patch("src.api.task_routes.create_pool") as mock_create_pool:
            mock_pool = AsyncMock()
            mock_job = AsyncMock()
            mock_pool.job.return_value = mock_job
            mock_create_pool.return_value = mock_pool

            response = client.post(f"/tasks/{task_id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

        # Verify task status updated
        task_repo = SQLAlchemyTaskRepository(db_session)
        updated_task = task_repo.find_by_id(task_id)
        assert updated_task.status == "cancelled"

    def test_cancel_completed_task_fails(self, client, db_session, sample_task):
        """Test that cancelling a completed task fails."""
        sample_task.status = "completed"
        sample_task.completed_at = datetime.utcnow()
        db_session.commit()

        task_id = sample_task.task_id
        response = client.post(f"/tasks/{task_id}/cancel")

        assert response.status_code == 400
        data = response.json()
        assert "cannot cancel" in data["detail"].lower()

    def test_cancel_failed_task_fails(self, client, db_session, sample_task):
        """Test that cancelling a failed task fails."""
        sample_task.status = "failed"
        sample_task.error = "Test error"
        db_session.commit()

        task_id = sample_task.task_id
        response = client.post(f"/tasks/{task_id}/cancel")

        assert response.status_code == 400
        data = response.json()
        assert "cannot cancel" in data["detail"].lower()

    def test_cancel_nonexistent_task(self, client):
        """Test cancelling a task that doesn't exist."""
        response = client.post("/tasks/nonexistent-task/cancel")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_cancel_task_redis_failure_still_marks_cancelled(
        self, client, db_session, sample_task
    ):
        """Test that task is marked cancelled even if Redis abort fails."""
        task_id = sample_task.task_id

        with patch("src.api.task_routes.create_pool") as mock_create_pool:
            # Simulate Redis failure
            mock_create_pool.side_effect = Exception("Redis connection failed")

            response = client.post(f"/tasks/{task_id}/cancel")

        # Should still succeed (soft cancellation)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

        # Verify task is marked cancelled in database
        task_repo = SQLAlchemyTaskRepository(db_session)
        updated_task = task_repo.find_by_id(task_id)
        assert updated_task.status == "cancelled"


class TestRetryTask:
    """Tests for POST /tasks/{task_id}/retry endpoint."""

    def test_retry_failed_task(self, client, db_session, sample_task, sample_video):
        """Test retrying a failed task."""
        sample_task.status = "failed"
        sample_task.error = "Test error"
        db_session.commit()

        task_id = sample_task.task_id

        with patch("src.api.task_routes.JobProducer") as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            mock_producer.enqueue_task.return_value = f"ml_{task_id}"

            response = client.post(f"/tasks/{task_id}/retry")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "pending"
        assert data["job_id"] == f"ml_{task_id}"

        # Verify task status reset
        task_repo = SQLAlchemyTaskRepository(db_session)
        updated_task = task_repo.find_by_id(task_id)
        assert updated_task.status == "pending"
        assert updated_task.error is None
        assert updated_task.started_at is None
        assert updated_task.completed_at is None

    def test_retry_cancelled_task(self, client, db_session, sample_task, sample_video):
        """Test retrying a cancelled task."""
        sample_task.status = "cancelled"
        db_session.commit()

        task_id = sample_task.task_id

        with patch("src.api.task_routes.JobProducer") as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            mock_producer.enqueue_task.return_value = f"ml_{task_id}"

            response = client.post(f"/tasks/{task_id}/retry")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"

    def test_retry_pending_task_fails(self, client, db_session, sample_task):
        """Test that retrying a pending task fails."""
        task_id = sample_task.task_id
        response = client.post(f"/tasks/{task_id}/retry")

        assert response.status_code == 400
        data = response.json()
        assert "cannot retry" in data["detail"].lower()

    def test_retry_running_task_fails(self, client, db_session, sample_task):
        """Test that retrying a running task fails."""
        sample_task.status = "running"
        sample_task.started_at = datetime.utcnow()
        db_session.commit()

        task_id = sample_task.task_id
        response = client.post(f"/tasks/{task_id}/retry")

        assert response.status_code == 400
        data = response.json()
        assert "cannot retry" in data["detail"].lower()

    def test_retry_completed_task_fails(self, client, db_session, sample_task):
        """Test that retrying a completed task fails."""
        sample_task.status = "completed"
        sample_task.completed_at = datetime.utcnow()
        db_session.commit()

        task_id = sample_task.task_id
        response = client.post(f"/tasks/{task_id}/retry")

        assert response.status_code == 400
        data = response.json()
        assert "cannot retry" in data["detail"].lower()

    def test_retry_nonexistent_task(self, client):
        """Test retrying a task that doesn't exist."""
        response = client.post("/tasks/nonexistent-task/retry")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestListTasks:
    """Tests for GET /tasks endpoint."""

    def test_list_all_tasks(self, client, db_session, sample_task):
        """Test listing all tasks."""
        response = client.get("/tasks")

        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert data["total"] >= 1

    def test_list_tasks_with_pagination(self, client, db_session, sample_task):
        """Test listing tasks with pagination."""
        response = client.get("/tasks?limit=5&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 0
        assert len(data["tasks"]) <= 5

    def test_list_tasks_filter_by_status(self, client, db_session, sample_task):
        """Test filtering tasks by status."""
        response = client.get("/tasks?status=pending")

        assert response.status_code == 200
        data = response.json()
        assert all(task["status"] == "pending" for task in data["tasks"])

    def test_list_tasks_filter_by_task_type(self, client, db_session, sample_task):
        """Test filtering tasks by task type."""
        response = client.get("/tasks?task_type=object_detection")

        assert response.status_code == 200
        data = response.json()
        assert all(task["task_type"] == "object_detection" for task in data["tasks"])

    def test_list_tasks_filter_by_video_id(self, client, db_session, sample_task):
        """Test filtering tasks by video ID."""
        video_id = sample_task.video_id
        response = client.get(f"/tasks?video_id={video_id}")

        assert response.status_code == 200
        data = response.json()
        assert all(task["video_id"] == video_id for task in data["tasks"])

    def test_list_tasks_sort_by_created_at_desc(self, client, db_session, sample_task):
        """Test sorting tasks by created_at descending."""
        response = client.get("/tasks?sort_by=created_at&sort_order=desc")

        assert response.status_code == 200
        data = response.json()
        tasks = data["tasks"]
        if len(tasks) > 1:
            # Verify descending order
            for i in range(len(tasks) - 1):
                assert tasks[i]["created_at"] >= tasks[i + 1]["created_at"]

    def test_list_tasks_sort_by_created_at_asc(self, client, db_session, sample_task):
        """Test sorting tasks by created_at ascending."""
        response = client.get("/tasks?sort_by=created_at&sort_order=asc")

        assert response.status_code == 200
        data = response.json()
        tasks = data["tasks"]
        if len(tasks) > 1:
            # Verify ascending order
            for i in range(len(tasks) - 1):
                assert tasks[i]["created_at"] <= tasks[i + 1]["created_at"]

    def test_list_tasks_invalid_status(self, client):
        """Test that invalid status filter returns error."""
        response = client.get("/tasks?status=invalid_status")

        assert response.status_code == 400
        data = response.json()
        assert "invalid status" in data["detail"].lower()

    def test_list_tasks_invalid_sort_by(self, client):
        """Test that invalid sort_by returns error."""
        response = client.get("/tasks?sort_by=invalid_field")

        assert response.status_code == 400
        data = response.json()
        assert "invalid sort_by" in data["detail"].lower()

    def test_list_tasks_invalid_sort_order(self, client):
        """Test that invalid sort_order returns error."""
        response = client.get("/tasks?sort_order=invalid")

        assert response.status_code == 400
        data = response.json()
        assert "invalid sort_order" in data["detail"].lower()

    def test_list_tasks_limit_capped_at_100(self, client, db_session, sample_task):
        """Test that limit is capped at 100."""
        response = client.get("/tasks?limit=200")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 100

    def test_list_tasks_negative_offset_defaults_to_zero(
        self, client, db_session, sample_task
    ):
        """Test that negative offset defaults to 0."""
        response = client.get("/tasks?offset=-5")

        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 0

    def test_list_tasks_response_format(self, client, db_session, sample_task):
        """Test that response has correct format."""
        response = client.get("/tasks")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert isinstance(data["tasks"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["limit"], int)
        assert isinstance(data["offset"], int)

        # Check task structure
        if data["tasks"]:
            task = data["tasks"][0]
            assert "task_id" in task
            assert "task_type" in task
            assert "status" in task
            assert "video_id" in task
            assert "created_at" in task
            assert "started_at" in task
            assert "completed_at" in task
            assert "error" in task
