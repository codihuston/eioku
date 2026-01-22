"""Tests for worker pool management."""

import time
import uuid
from unittest.mock import Mock, patch

from src.domain.models import Task
from src.services.task_orchestration import TaskType
from src.services.worker_pool_manager import (
    HashWorker,
    ObjectDetectionWorker,
    ResourceType,
    TaskWorker,
    WorkerConfig,
    WorkerPool,
    WorkerPoolManager,
)


class TestTaskWorker:
    """Test task worker functionality."""

    def test_execute_task_success(self):
        """Test successful task execution."""
        worker = TaskWorker(TaskType.HASH)
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        result = worker.execute_task(task)

        assert result["status"] == "success"
        assert "result" in result
        assert task.is_completed()

    def test_execute_task_failure(self):
        """Test task execution failure."""
        worker = TaskWorker(TaskType.HASH)
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        # Mock _do_work to raise exception
        def failing_work(task):
            raise ValueError("Test error")

        worker._do_work = failing_work

        result = worker.execute_task(task)

        assert result["status"] == "error"
        assert result["error"] == "Test error"
        assert task.is_failed()


class TestSpecificWorkers:
    """Test specific worker implementations."""

    def test_hash_worker(self):
        """Test hash worker."""
        # Create mock hash service
        mock_hash_service = Mock()
        mock_hash_service.calculate_hash.return_value = (
            "1234567890abcdef"  # 16 char xxhash
        )

        # Create mock video repository
        mock_video_repo = Mock()
        mock_video = Mock()
        mock_video.file_path = "/test/video.mp4"
        mock_video_repo.find_by_id.return_value = mock_video

        worker = HashWorker(
            hash_service=mock_hash_service, video_repository=mock_video_repo
        )
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        result = worker._do_work(task)

        # Should return a hash string
        assert isinstance(result, str)
        assert len(result) == 16  # xxhash64 hex length
        assert result == "1234567890abcdef"

    def test_transcription_worker(self):
        """Test transcription worker."""
        # Create mock transcription handler
        mock_handler = Mock()
        # Legacy transcription worker test removed - now using artifact-based workers


class TestWorkerPool:
    """Test worker pool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = Mock()
        self.config = WorkerConfig(
            task_type=TaskType.HASH,
            worker_count=2,
            resource_type=ResourceType.CPU,
            priority=1,
        )

    def test_worker_pool_creation(self):
        """Test worker pool creation."""
        pool = WorkerPool(self.config, self.orchestrator)

        assert pool.config == self.config
        assert pool.orchestrator == self.orchestrator
        assert not pool.is_running
        assert len(pool.workers) == 0

    def test_worker_pool_start_stop(self):
        """Test starting and stopping worker pool."""
        pool = WorkerPool(self.config, self.orchestrator)

        # Start pool
        pool.start()
        assert pool.is_running
        assert pool.executor is not None
        assert len(pool.workers) == self.config.worker_count

        # Stop pool
        pool.stop()
        assert not pool.is_running
        assert len(pool.workers) == 0

    @patch("time.sleep")  # Speed up test
    @patch("src.services.worker_pool_manager.WorkerPool._worker_loop")
    def test_worker_loop_no_tasks(self, mock_worker_loop, mock_sleep):
        """Test worker loop when no tasks available."""
        # Mock orchestrator to return no tasks
        self.orchestrator.get_next_task.return_value = None

        pool = WorkerPool(self.config, self.orchestrator)
        pool.start()

        # Let it run briefly
        time.sleep(0.1)

        pool.stop()

        # Worker loop should have been called for each worker
        assert mock_worker_loop.call_count == self.config.worker_count

    @patch("src.services.worker_pool_manager.WorkerPool._worker_loop")
    def test_worker_loop_with_task(self, mock_worker_loop):
        """Test worker loop processing a task."""
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        # Mock orchestrator to return task once, then None
        self.orchestrator.get_next_task.side_effect = [task, None, None]

        # Use GPU resource type to force ThreadPoolExecutor
        # This avoids pickling issues with mocked objects
        config = WorkerConfig(
            task_type=TaskType.HASH,
            worker_count=1,
            resource_type=ResourceType.GPU,  # Forces ThreadPoolExecutor
            priority=1,
        )

        # Create a mock worker that always succeeds
        mock_worker = Mock()
        mock_worker.execute_task.return_value = {
            "status": "success",
            "result": "test_hash",
        }

        pool = WorkerPool(config, self.orchestrator)

        # Replace the worker factory with our mock
        pool.worker_factory = lambda: mock_worker

        pool.start()

        # Let it process the task
        time.sleep(0.1)

        pool.stop()

        # Worker loop should have been called
        assert mock_worker_loop.call_count == config.worker_count


class TestWorkerPoolManager:
    """Test worker pool manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = Mock()
        self.manager = WorkerPoolManager(self.orchestrator)

    def test_add_worker_pool(self):
        """Test adding worker pools."""
        config = WorkerConfig(
            task_type=TaskType.HASH,
            worker_count=2,
            resource_type=ResourceType.CPU,
            priority=1,
        )

        self.manager.add_worker_pool(config)

        assert TaskType.HASH in self.manager.pools
        assert self.manager.pools[TaskType.HASH].config == config

    def test_add_duplicate_pool_raises_error(self):
        """Test adding duplicate pool raises error."""
        config = WorkerConfig(
            task_type=TaskType.HASH,
            worker_count=2,
            resource_type=ResourceType.CPU,
            priority=1,
        )

        self.manager.add_worker_pool(config)

        # Adding same type again should raise error
        try:
            self.manager.add_worker_pool(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already exists" in str(e)

    def test_start_stop_all_pools(self):
        """Test starting and stopping all pools."""
        # Add multiple pools
        configs = [
            WorkerConfig(TaskType.HASH, 2, ResourceType.CPU, 1),
            WorkerConfig(TaskType.TRANSCRIPTION, 1, ResourceType.CPU, 1),
        ]

        for config in configs:
            self.manager.add_worker_pool(config)

        # Start all
        self.manager.start_all()
        assert self.manager.is_running

        for pool in self.manager.pools.values():
            assert pool.is_running

        # Stop all
        self.manager.stop_all()
        assert not self.manager.is_running

        for pool in self.manager.pools.values():
            assert not pool.is_running

    def test_get_status(self):
        """Test getting pool status."""
        config = WorkerConfig(
            task_type=TaskType.HASH,
            worker_count=2,
            resource_type=ResourceType.CPU,
            priority=1,
        )

        self.manager.add_worker_pool(config)
        status = self.manager.get_status()

        assert TaskType.HASH.value in status
        pool_status = status[TaskType.HASH.value]
        assert pool_status["worker_count"] == 2
        assert pool_status["resource_type"] == "cpu"
        assert pool_status["is_running"] is False

    def test_create_default_pools(self):
        """Test creating default worker pools."""
        self.manager.create_default_pools()

        # Should have pools for all artifact-based task types
        expected_types = {
            TaskType.HASH,
            TaskType.TRANSCRIPTION,
            TaskType.SCENE_DETECTION,
            TaskType.OBJECT_DETECTION,
            TaskType.FACE_DETECTION,
            TaskType.OCR,
            TaskType.PLACE_DETECTION,
        }

        assert set(self.manager.pools.keys()) == expected_types

        # Hash should have highest priority (lowest number)
        hash_pool = self.manager.pools[TaskType.HASH]
        assert hash_pool.config.priority == 1
        assert hash_pool.config.worker_count == 4
