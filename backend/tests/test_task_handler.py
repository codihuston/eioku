"""Tests for Worker Service task handler."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.models import Task
from src.workers.task_handler import (
    EXPECTED_ARTIFACTS,
    poll_for_artifacts,
    process_ml_task,
)


class TestProcessMLTask:
    """Test process_ml_task handler."""

    @pytest.mark.asyncio
    async def test_process_ml_task_success(self):
        """Test successful task processing."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"
        video_path = "/path/to/video.mp4"
        config = {"confidence": 0.5}

        # Mock repositories
        mock_task_repo = MagicMock()
        mock_artifact_repo = MagicMock()

        # Create mock task
        mock_task = Task(
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
            status="pending",
            priority=5,
            dependencies=[],
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            error=None,
        )

        mock_task_repo.find_by_video_and_type.return_value = [mock_task]

        # Make update return a new task with updated status
        def update_side_effect(task):
            return Task(
                task_id=task.task_id,
                video_id=task.video_id,
                task_type=task.task_type,
                status=task.status,
                priority=task.priority,
                dependencies=task.dependencies,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                error=task.error,
            )

        mock_task_repo.update.side_effect = update_side_effect

        # Mock job producer
        mock_job_producer = AsyncMock()
        mock_job_producer.enqueue_to_ml_jobs.return_value = "ml_task_123"

        # Mock polling
        with patch("src.workers.task_handler.get_db") as mock_get_db, patch(
            "src.workers.task_handler.SQLAlchemyTaskRepository",
            return_value=mock_task_repo,
        ), patch(
            "src.workers.task_handler.SqlArtifactRepository",
            return_value=mock_artifact_repo,
        ), patch(
            "src.workers.task_handler.JobProducer",
            return_value=mock_job_producer,
        ), patch(
            "src.workers.task_handler.poll_for_artifacts",
            return_value=1,
        ):
            mock_session = MagicMock()
            mock_get_db.return_value = iter([mock_session])

            result = await process_ml_task(
                task_id=task_id,
                task_type=task_type,
                video_id=video_id,
                video_path=video_path,
                config=config,
            )

            # Verify result
            assert result["task_id"] == task_id
            assert result["status"] == "completed"
            assert result["artifact_count"] == 1

            # Verify task status was updated twice (RUNNING then COMPLETED)
            assert mock_task_repo.update.call_count == 2

            # Verify job was enqueued to ml_jobs
            mock_job_producer.enqueue_to_ml_jobs.assert_called_once_with(
                task_id=task_id,
                task_type=task_type,
                video_id=video_id,
                video_path=video_path,
                config=config,
            )

    @pytest.mark.asyncio
    async def test_process_ml_task_task_not_found(self):
        """Test error when task not found."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"
        video_path = "/path/to/video.mp4"

        mock_task_repo = MagicMock()
        mock_task_repo.find_by_video_and_type.return_value = []

        with patch("src.workers.task_handler.get_db") as mock_get_db, patch(
            "src.workers.task_handler.SQLAlchemyTaskRepository",
            return_value=mock_task_repo,
        ):
            mock_session = MagicMock()
            mock_get_db.return_value = iter([mock_session])

            with pytest.raises(ValueError, match="Task not found"):
                await process_ml_task(
                    task_id=task_id,
                    task_type=task_type,
                    video_id=video_id,
                    video_path=video_path,
                )

    @pytest.mark.asyncio
    async def test_process_ml_task_already_completed(self):
        """Test error when task is already completed."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"
        video_path = "/path/to/video.mp4"

        mock_task_repo = MagicMock()

        # Create completed task
        completed_task = Task(
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
            status="completed",
            priority=5,
            dependencies=[],
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            error=None,
        )

        mock_task_repo.find_by_video_and_type.return_value = [completed_task]

        with patch("src.workers.task_handler.get_db") as mock_get_db, patch(
            "src.workers.task_handler.SQLAlchemyTaskRepository",
            return_value=mock_task_repo,
        ):
            mock_session = MagicMock()
            mock_get_db.return_value = iter([mock_session])

            with pytest.raises(ValueError, match="Cannot process task"):
                await process_ml_task(
                    task_id=task_id,
                    task_type=task_type,
                    video_id=video_id,
                    video_path=video_path,
                )

    @pytest.mark.asyncio
    async def test_process_ml_task_cancellation(self):
        """Test task cancellation handling."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"
        video_path = "/path/to/video.mp4"

        mock_task_repo = MagicMock()

        pending_task = Task(
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
            status="pending",
            priority=5,
            dependencies=[],
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            error=None,
        )

        mock_task_repo.find_by_video_and_type.return_value = [pending_task]

        with patch("src.workers.task_handler.get_db") as mock_get_db, patch(
            "src.workers.task_handler.SQLAlchemyTaskRepository",
            return_value=mock_task_repo,
        ), patch("src.workers.task_handler.SqlArtifactRepository"), patch(
            "src.workers.task_handler.JobProducer"
        ) as mock_job_producer_class, patch(
            "src.workers.task_handler.poll_for_artifacts",
            side_effect=asyncio.CancelledError(),
        ):
            mock_session = MagicMock()
            mock_get_db.return_value = iter([mock_session])

            mock_job_producer = AsyncMock()
            mock_job_producer_class.return_value = mock_job_producer

            with pytest.raises(asyncio.CancelledError):
                await process_ml_task(
                    task_id=task_id,
                    task_type=task_type,
                    video_id=video_id,
                    video_path=video_path,
                )

            # Verify task was marked as cancelled
            cancelled_call = [
                call
                for call in mock_task_repo.update.call_args_list
                if call[0][0].status == "cancelled"
            ]
            assert len(cancelled_call) > 0

    @pytest.mark.asyncio
    async def test_process_ml_task_exception_handling(self):
        """Test exception handling and task marked as failed."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"
        video_path = "/path/to/video.mp4"

        mock_task_repo = MagicMock()

        pending_task = Task(
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
            status="pending",
            priority=5,
            dependencies=[],
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            error=None,
        )

        mock_task_repo.find_by_video_and_type.return_value = [pending_task]

        with patch("src.workers.task_handler.get_db") as mock_get_db, patch(
            "src.workers.task_handler.SQLAlchemyTaskRepository",
            return_value=mock_task_repo,
        ), patch("src.workers.task_handler.SqlArtifactRepository"), patch(
            "src.workers.task_handler.JobProducer"
        ) as mock_job_producer_class:
            mock_session = MagicMock()
            mock_get_db.return_value = iter([mock_session])

            mock_job_producer = AsyncMock()
            mock_job_producer.enqueue_to_ml_jobs.side_effect = RuntimeError(
                "Redis connection failed"
            )
            mock_job_producer_class.return_value = mock_job_producer

            with pytest.raises(RuntimeError):
                await process_ml_task(
                    task_id=task_id,
                    task_type=task_type,
                    video_id=video_id,
                    video_path=video_path,
                )

            # Verify task was marked as failed
            failed_call = [
                call
                for call in mock_task_repo.update.call_args_list
                if call[0][0].status == "failed"
            ]
            assert len(failed_call) > 0


class TestPollForArtifacts:
    """Test artifact polling logic."""

    @pytest.mark.asyncio
    async def test_poll_for_artifacts_success(self):
        """Test successful artifact polling."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"

        mock_artifact_repo = MagicMock()
        mock_session = MagicMock()

        # Mock artifacts found
        mock_artifacts = [MagicMock(), MagicMock()]
        mock_artifact_repo.get_by_asset.return_value = mock_artifacts

        result = await poll_for_artifacts(
            task_id=task_id,
            task_type=task_type,
            video_id=video_id,
            session=mock_session,
            artifact_repo=mock_artifact_repo,
            initial_delay=0.01,
            max_delay=0.1,
            timeout=5.0,
        )

        assert result == 2
        mock_artifact_repo.get_by_asset.assert_called_once_with(
            asset_id=video_id,
            artifact_type=task_type,
        )

    @pytest.mark.asyncio
    async def test_poll_for_artifacts_timeout(self):
        """Test polling timeout."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"

        mock_artifact_repo = MagicMock()
        mock_session = MagicMock()

        # Mock no artifacts found
        mock_artifact_repo.get_by_asset.return_value = []

        with pytest.raises(TimeoutError, match="Polling timeout exceeded"):
            await poll_for_artifacts(
                task_id=task_id,
                task_type=task_type,
                video_id=video_id,
                session=mock_session,
                artifact_repo=mock_artifact_repo,
                initial_delay=0.01,
                max_delay=0.01,
                timeout=0.05,
            )

    @pytest.mark.asyncio
    async def test_poll_for_artifacts_invalid_task_type(self):
        """Test error with invalid task type."""
        task_id = "task_123"
        task_type = "invalid_type"
        video_id = "video_456"

        mock_artifact_repo = MagicMock()
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Unknown task type"):
            await poll_for_artifacts(
                task_id=task_id,
                task_type=task_type,
                video_id=video_id,
                session=mock_session,
                artifact_repo=mock_artifact_repo,
            )

    @pytest.mark.asyncio
    async def test_poll_for_artifacts_missing_video_id(self):
        """Test error when video_id is missing."""
        task_id = "task_123"
        task_type = "object_detection"

        mock_artifact_repo = MagicMock()
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="video_id is required"):
            await poll_for_artifacts(
                task_id=task_id,
                task_type=task_type,
                video_id=None,
                session=mock_session,
                artifact_repo=mock_artifact_repo,
            )

    @pytest.mark.asyncio
    async def test_poll_for_artifacts_exponential_backoff(self):
        """Test exponential backoff during polling."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"

        mock_artifact_repo = MagicMock()
        mock_session = MagicMock()

        # Return empty first 2 times, then return artifacts
        mock_artifacts = [MagicMock()]
        mock_artifact_repo.get_by_asset.side_effect = [[], [], mock_artifacts]

        result = await poll_for_artifacts(
            task_id=task_id,
            task_type=task_type,
            video_id=video_id,
            session=mock_session,
            artifact_repo=mock_artifact_repo,
            initial_delay=0.01,
            max_delay=0.1,
            timeout=5.0,
        )

        assert result == 1
        assert mock_artifact_repo.get_by_asset.call_count == 3

    @pytest.mark.asyncio
    async def test_poll_for_artifacts_query_error_recovery(self):
        """Test recovery from query errors during polling."""
        task_id = "task_123"
        task_type = "object_detection"
        video_id = "video_456"

        mock_artifact_repo = MagicMock()
        mock_session = MagicMock()

        # Raise error first time, then return artifacts
        mock_artifacts = [MagicMock()]
        mock_artifact_repo.get_by_asset.side_effect = [
            RuntimeError("Database error"),
            mock_artifacts,
        ]

        result = await poll_for_artifacts(
            task_id=task_id,
            task_type=task_type,
            video_id=video_id,
            session=mock_session,
            artifact_repo=mock_artifact_repo,
            initial_delay=0.01,
            max_delay=0.1,
            timeout=5.0,
        )

        assert result == 1
        assert mock_artifact_repo.get_by_asset.call_count == 2


class TestExpectedArtifacts:
    """Test expected artifact counts."""

    def test_expected_artifacts_all_task_types(self):
        """Test that all task types have expected artifact counts."""
        expected_types = {
            "object_detection",
            "face_detection",
            "transcription",
            "ocr",
            "place_detection",
            "scene_detection",
        }

        for task_type in expected_types:
            assert task_type in EXPECTED_ARTIFACTS
            assert EXPECTED_ARTIFACTS[task_type] == 1
