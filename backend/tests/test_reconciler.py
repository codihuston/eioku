"""Unit tests for Reconciler."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.models import Task
from src.workers.reconciler import LONG_RUNNING_THRESHOLD, Reconciler


class TestReconcilerInitialization:
    """Test Reconciler initialization."""

    def test_reconciler_init_with_session(self):
        """Test Reconciler initialization with provided session."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        assert reconciler.session == mock_session
        assert reconciler._owns_session is False

    def test_reconciler_init_without_session(self):
        """Test Reconciler initialization without session."""
        reconciler = Reconciler()

        assert reconciler.session is None
        assert reconciler._owns_session is True


class TestSyncPendingTasks:
    """Test PENDING task synchronization."""

    @pytest.mark.asyncio
    async def test_sync_pending_tasks_all_have_jobs(self):
        """Test syncing PENDING tasks when all have jobs in Redis."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock tasks
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="pending",
        )
        task2 = Task(
            task_id="task-2",
            video_id="video-2",
            task_type="face_detection",
            status="pending",
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1, task2]
        reconciler.task_repo = mock_task_repo

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock job existence check
        with patch.object(reconciler, "_check_job_exists") as mock_check:
            mock_check.return_value = True

            stats = await reconciler._sync_pending_tasks()

            assert stats["checked"] == 2
            assert stats["reenqueued"] == 0
            mock_check.assert_called()

    @pytest.mark.asyncio
    async def test_sync_pending_tasks_missing_jobs(self):
        """Test syncing PENDING tasks when jobs are missing in Redis."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock tasks
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="pending",
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock job existence check - job doesn't exist
        with patch.object(reconciler, "_check_job_exists") as mock_check:
            mock_check.return_value = False

            stats = await reconciler._sync_pending_tasks()

            assert stats["checked"] == 1
            assert stats["reenqueued"] == 1
            mock_job_producer.enqueue_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_pending_tasks_error_handling(self):
        """Test error handling in PENDING task sync."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock task
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="pending",
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock job existence check to raise error
        with patch.object(reconciler, "_check_job_exists") as mock_check:
            mock_check.side_effect = Exception("Redis error")

            stats = await reconciler._sync_pending_tasks()

            # Should continue despite error
            assert stats["checked"] == 1
            assert stats["reenqueued"] == 0


class TestSyncRunningTasks:
    """Test RUNNING task synchronization."""

    @pytest.mark.asyncio
    async def test_sync_running_tasks_job_missing(self):
        """Test syncing RUNNING tasks when job is missing in Redis."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock task
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="running",
            started_at=datetime.utcnow(),
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock job existence check - job doesn't exist
        with patch.object(reconciler, "_check_job_exists") as mock_check:
            mock_check.return_value = False

            stats = await reconciler._sync_running_tasks()

            assert stats["checked"] == 1
            assert stats["synced"] == 1
            # Task should be reset to PENDING
            mock_task_repo.update.assert_called()
            updated_task = mock_task_repo.update.call_args[0][0]
            assert updated_task.status == "pending"

    @pytest.mark.asyncio
    async def test_sync_running_tasks_job_complete(self):
        """Test syncing RUNNING tasks when job is complete in Redis."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock task
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="running",
            started_at=datetime.utcnow(),
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock job checks
        with patch.object(reconciler, "_check_job_exists") as mock_check, patch.object(
            reconciler, "_get_job_status"
        ) as mock_status:
            mock_check.return_value = True
            mock_status.return_value = "complete"

            stats = await reconciler._sync_running_tasks()

            assert stats["checked"] == 1
            assert stats["synced"] == 1
            # Task should be updated to COMPLETED
            mock_task_repo.update.assert_called()
            updated_task = mock_task_repo.update.call_args[0][0]
            assert updated_task.status == "completed"

    @pytest.mark.asyncio
    async def test_sync_running_tasks_job_failed(self):
        """Test syncing RUNNING tasks when job failed in Redis."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock task
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="running",
            started_at=datetime.utcnow(),
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock job checks
        with patch.object(reconciler, "_check_job_exists") as mock_check, patch.object(
            reconciler, "_get_job_status"
        ) as mock_status:
            mock_check.return_value = True
            mock_status.return_value = "failed"

            stats = await reconciler._sync_running_tasks()

            assert stats["checked"] == 1
            assert stats["synced"] == 1
            # Task should be updated to FAILED
            mock_task_repo.update.assert_called()
            updated_task = mock_task_repo.update.call_args[0][0]
            assert updated_task.status == "failed"


class TestAlertLongRunningTasks:
    """Test long-running task alerting."""

    @pytest.mark.asyncio
    async def test_alert_long_running_tasks_none(self):
        """Test alerting when no tasks are long-running."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock task that's not long-running
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="running",
            started_at=datetime.utcnow() - timedelta(seconds=60),
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        stats = await reconciler._alert_long_running_tasks()

        assert stats["alerted"] == 0

    @pytest.mark.asyncio
    async def test_alert_long_running_tasks_found(self):
        """Test alerting when long-running tasks are found."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock task that's long-running
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="running",
            started_at=datetime.utcnow()
            - timedelta(seconds=LONG_RUNNING_THRESHOLD + 100),
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        stats = await reconciler._alert_long_running_tasks()

        assert stats["alerted"] == 1

    @pytest.mark.asyncio
    async def test_alert_long_running_tasks_no_start_time(self):
        """Test alerting when task has no start time."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Create mock task with no start time
        task1 = Task(
            task_id="task-1",
            video_id="video-1",
            task_type="object_detection",
            status="running",
            started_at=None,
        )

        # Mock task repository
        mock_task_repo = MagicMock()
        mock_task_repo.find_by_status.return_value = [task1]
        reconciler.task_repo = mock_task_repo

        stats = await reconciler._alert_long_running_tasks()

        assert stats["alerted"] == 0


class TestReconcilerRun:
    """Test main reconciler run."""

    @pytest.mark.asyncio
    async def test_reconciler_run_success(self):
        """Test successful reconciler run."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock sync methods
        with patch.object(
            reconciler, "_sync_pending_tasks"
        ) as mock_pending, patch.object(
            reconciler, "_sync_running_tasks"
        ) as mock_running, patch.object(
            reconciler, "_alert_long_running_tasks"
        ) as mock_alert, patch(
            "src.workers.reconciler.JobProducer"
        ) as mock_producer_class:
            mock_producer_class.return_value = mock_job_producer
            mock_pending.return_value = {"checked": 5, "reenqueued": 1}
            mock_running.return_value = {"checked": 3, "synced": 1}
            mock_alert.return_value = {"alerted": 0}

            stats = await reconciler.run()

            assert stats["pending_checked"] == 5
            assert stats["pending_reenqueued"] == 1
            assert stats["running_checked"] == 3
            assert stats["running_synced"] == 1
            assert stats["long_running_alerted"] == 0
            assert len(stats["errors"]) == 0

    @pytest.mark.asyncio
    async def test_reconciler_run_with_errors(self):
        """Test reconciler run with errors."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        # Mock job producer
        mock_job_producer = AsyncMock()
        reconciler.job_producer = mock_job_producer

        # Mock sync methods with errors
        with patch.object(
            reconciler, "_sync_pending_tasks"
        ) as mock_pending, patch.object(
            reconciler, "_sync_running_tasks"
        ) as mock_running, patch.object(
            reconciler, "_alert_long_running_tasks"
        ) as mock_alert, patch(
            "src.workers.reconciler.JobProducer"
        ) as mock_producer_class:
            mock_producer_class.return_value = mock_job_producer
            mock_pending.side_effect = Exception("Pending sync error")
            mock_running.return_value = {"checked": 3, "synced": 1}
            mock_alert.return_value = {"alerted": 0}

            stats = await reconciler.run()

            assert len(stats["errors"]) == 1
            assert "Pending sync error" in stats["errors"][0]


class TestCheckJobExists:
    """Test job existence check."""

    @pytest.mark.asyncio
    async def test_check_job_exists_found(self):
        """Test checking when job exists in Redis."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.get.return_value = '{"status": "in_progress"}'

            result = await reconciler._check_job_exists("task-1")

            assert result is True

    @pytest.mark.asyncio
    async def test_check_job_exists_not_found(self):
        """Test checking when job doesn't exist in Redis."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.get.return_value = None

            result = await reconciler._check_job_exists("task-1")

            assert result is False

    @pytest.mark.asyncio
    async def test_check_job_exists_error(self):
        """Test checking when Redis error occurs."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.side_effect = Exception("Redis connection error")

            result = await reconciler._check_job_exists("task-1")

            # Should return True on error to avoid re-enqueueing
            assert result is True


class TestGetJobStatus:
    """Test job status retrieval."""

    @pytest.mark.asyncio
    async def test_get_job_status_found(self):
        """Test getting job status when job exists."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.get.return_value = '{"status": "complete"}'

            result = await reconciler._get_job_status("task-1")

            # Currently returns None (simplified implementation)
            assert result is None

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self):
        """Test getting job status when job doesn't exist."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.get.return_value = None

            result = await reconciler._get_job_status("task-1")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_job_status_error(self):
        """Test getting job status when Redis error occurs."""
        mock_session = MagicMock()
        reconciler = Reconciler(session=mock_session)

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.side_effect = Exception("Redis connection error")

            result = await reconciler._get_job_status("task-1")

            assert result is None
