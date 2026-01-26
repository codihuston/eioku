"""Unit tests for non-blocking Redis queue pattern.

Tests the two-phase worker pattern:
1. process_ml_task() - sends to ML Service (fire-and-forget)
2. process_ml_results() - listens on Redis and processes artifacts
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workers.task_handler import process_ml_task


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock()
    session.close = MagicMock()
    return session


@pytest.fixture
def mock_task_repo():
    """Create a mock task repository."""
    repo = MagicMock()
    task_obj = MagicMock()
    task_obj.status = "pending"
    repo.find_by_video_and_type.return_value = [task_obj]
    repo.update = MagicMock()
    return repo


@pytest.fixture
def mock_artifact_repo():
    """Create a mock artifact repository."""
    repo = MagicMock()
    repo.create = MagicMock()
    return repo


@pytest.mark.asyncio
async def test_process_ml_task_sends_to_ml_service_and_returns_immediately(
    mock_session, mock_task_repo
):
    """Test that process_ml_task sends to ML Service and returns immediately."""
    task_id = "test_task_123"
    task_type = "object_detection"
    video_id = "video_456"
    video_path = "/path/to/video.mp4"

    with patch("src.workers.task_handler.get_db") as mock_get_db:
        mock_get_db.return_value = iter([mock_session])

        with patch(
            "src.workers.task_handler.SQLAlchemyTaskRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_task_repo

            with patch(
                "src.repositories.video_repository.VideoRepository"
            ) as mock_video_repo_class:
                mock_video_repo = MagicMock()
                mock_video = MagicMock()
                mock_video.file_hash = "abc123def456"
                mock_video_repo.find_by_id.return_value = mock_video
                mock_video_repo_class.return_value = mock_video_repo

                with patch(
                    "src.workers.task_handler.JobProducer"
                ) as mock_producer_class:
                    mock_producer = AsyncMock()
                    mock_producer.enqueue_to_ml_jobs = AsyncMock(return_value="job_123")
                    mock_producer_class.return_value = mock_producer

                    # Execute
                    result = await process_ml_task(
                        task_id=task_id,
                        task_type=task_type,
                        video_id=video_id,
                        video_path=video_path,
                        config={},
                    )

                    # Verify
                    assert result["task_id"] == task_id
                    assert result["status"] == "sent_to_ml_service"

                    # Verify task status was updated to RUNNING
                    assert mock_task_repo.update.called
                    updated_task = mock_task_repo.update.call_args[0][0]
                    assert updated_task.status == "running"

                    # Verify job was enqueued
                    assert mock_producer.enqueue_to_ml_jobs.called
