"""Tests for task orchestrator."""

import uuid
from datetime import datetime
from unittest.mock import Mock

from src.domain.models import Task, Video
from src.services.task_orchestration import TaskDependencyManager, TaskQueues, TaskType
from src.services.task_orchestrator import TaskOrchestrator


class TestTaskOrchestrator:
    """Test task orchestrator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.task_repo = Mock()
        self.video_repo = Mock()
        self.task_queues = TaskQueues()
        self.dependency_manager = TaskDependencyManager()

        self.orchestrator = TaskOrchestrator(
            task_repository=self.task_repo,
            video_repository=self.video_repo,
            task_queues=self.task_queues,
            dependency_manager=self.dependency_manager,
        )

    def test_create_hash_task_for_discovered_video(self):
        """Test creating hash task for discovered video."""
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="discovered",
            file_hash=None,
            file_size=1000,
        )

        # Mock repository responses
        self.task_repo.find_by_video_and_type.return_value = []
        self.task_repo.save.side_effect = lambda task: task

        # Create tasks
        tasks = self.orchestrator.create_tasks_for_video(video)

        # Should create hash task
        assert len(tasks) == 1
        assert tasks[0].task_type == TaskType.HASH.value
        assert tasks[0].video_id == video.video_id

        # Should be enqueued
        assert self.task_queues.get_queue_size(TaskType.HASH) == 1

    def test_create_parallel_tasks_for_hashed_video(self):
        """Test creating parallel tasks for hashed video."""
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="hashed",
            file_hash="abc123",
            file_size=1000,
        )

        # Mock repository responses
        self.task_repo.find_by_video_and_type.return_value = []
        self.task_repo.save.side_effect = lambda task: task

        # Create tasks
        tasks = self.orchestrator.create_tasks_for_video(video)

        # Should create parallel tasks
        expected_types = {
            TaskType.TRANSCRIPTION,
            TaskType.SCENE_DETECTION,
            TaskType.OBJECT_DETECTION,
            TaskType.FACE_DETECTION,
            TaskType.OCR,
            TaskType.PLACE_DETECTION,
        }
        created_types = {TaskType(t.task_type) for t in tasks}

        assert expected_types == created_types

        # All should be enqueued
        for task_type in expected_types:
            assert self.task_queues.get_queue_size(task_type) == 1

    def test_no_tasks_created_for_completed_video(self):
        """Test no tasks created for already completed video."""
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="completed",
            file_hash="abc123",
            file_size=1000,
        )

        # Mock repository responses - simulate all tasks already exist
        def mock_find_by_video_and_type(video_id, task_type):
            # Return existing completed task for any task type
            return [
                Task(
                    task_id=str(uuid.uuid4()),
                    video_id=video_id,
                    task_type=task_type,
                    status="completed",
                )
            ]

        self.task_repo.find_by_video_and_type.side_effect = mock_find_by_video_and_type

        # Create tasks
        tasks = self.orchestrator.create_tasks_for_video(video)

        # Should create no tasks since all already exist and are completed
        assert len(tasks) == 0

    def test_process_discovered_videos(self):
        """Test processing all discovered videos."""
        videos = [
            Video(
                video_id=str(uuid.uuid4()),
                file_path="/test/test1.mp4",
                filename="test1.mp4",
                last_modified=datetime.utcnow(),
                status="discovered",
                file_hash=None,
                file_size=1000,
            ),
            Video(
                video_id=str(uuid.uuid4()),
                file_path="/test/test2.mp4",
                filename="test2.mp4",
                last_modified=datetime.utcnow(),
                status="discovered",
                file_hash=None,
                file_size=2000,
            ),
        ]

        # Mock repository responses
        self.video_repo.find_by_status.return_value = videos
        self.task_repo.find_by_video_and_type.return_value = []
        self.task_repo.save.side_effect = lambda task: task

        # Process discovered videos
        count = self.orchestrator.process_discovered_videos()

        # Should create 2 hash tasks
        assert count == 2
        assert self.task_queues.get_queue_size(TaskType.HASH) == 2

    def test_handle_hash_task_completion(self):
        """Test handling hash task completion."""
        video_id = str(uuid.uuid4())
        video = Video(
            video_id=video_id,
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="discovered",
            file_hash=None,
            file_size=1000,
        )

        hash_task = Task(
            task_id=str(uuid.uuid4()),
            video_id=video_id,
            task_type=TaskType.HASH.value,
            status="running",
        )

        # Mock repository responses
        self.video_repo.find_by_id.return_value = video
        self.task_repo.find_by_video_and_type.return_value = []
        # Add this for completion check
        self.task_repo.find_by_video_id.return_value = []
        self.task_repo.save.side_effect = lambda task: task
        self.task_repo.update.return_value = hash_task
        self.video_repo.update.return_value = video

        # Handle completion
        self.orchestrator.handle_task_completion(hash_task)

        # Should mark task as completed
        assert hash_task.is_completed()

        # Should update video status to hashed
        self.video_repo.save.assert_called()

        # Should create parallel tasks (but video status needs to be
        # updated first) In real scenario, video would be reloaded with
        # hashed status

    def test_handle_task_failure(self):
        """Test handling task failure."""
        video_id = str(uuid.uuid4())
        video = Video(
            video_id=video_id,
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="discovered",
            file_hash=None,
            file_size=1000,
        )

        hash_task = Task(
            task_id=str(uuid.uuid4()),
            video_id=video_id,
            task_type=TaskType.HASH.value,
            status="running",
        )

        # Mock repository responses
        self.video_repo.find_by_id.return_value = video
        self.task_repo.update.return_value = hash_task
        self.video_repo.save.return_value = video

        # Handle failure
        error_msg = "File not found"
        self.orchestrator.handle_task_failure(hash_task, error_msg)

        # Should mark task as failed
        assert hash_task.is_failed()
        assert hash_task.error == error_msg

        # Should update video status to failed for hash task
        self.video_repo.save.assert_called()

    def test_get_next_task(self):
        """Test getting next task from queue."""
        # Create and enqueue a task
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        self.task_queues.enqueue(task)

        # Get next task
        next_task = self.orchestrator.get_next_task(TaskType.HASH)

        assert next_task is not None
        assert next_task.task_id == task.task_id

        # Queue should be empty now
        assert self.task_queues.get_queue_size(TaskType.HASH) == 0

    def test_retry_failed_tasks(self):
        """Test retrying failed tasks."""
        failed_task = Task(
            task_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="failed",
            error="Some error",
        )

        # Mock repository responses
        self.task_repo.find_by_status.return_value = [failed_task]
        self.task_repo.update.return_value = failed_task

        # Retry failed tasks
        count = self.orchestrator.retry_failed_tasks()

        # Should retry 1 task
        assert count == 1

        # Task should be reset
        assert failed_task.status == "pending"
        assert failed_task.error is None

        # Should be re-enqueued
        assert self.task_queues.get_queue_size(TaskType.HASH) == 1

    def test_get_queue_status(self):
        """Test getting queue status."""
        # Enqueue some tasks
        for i in range(3):
            task = Task(
                task_id=str(uuid.uuid4()),
                video_id=str(uuid.uuid4()),
                task_type=TaskType.HASH.value,
                status="pending",
            )
            self.task_queues.enqueue(task)

        for i in range(2):
            task = Task(
                task_id=str(uuid.uuid4()),
                video_id=str(uuid.uuid4()),
                task_type=TaskType.TRANSCRIPTION.value,
                status="pending",
            )
            self.task_queues.enqueue(task)

        # Get status
        status = self.orchestrator.get_queue_status()

        assert status[TaskType.HASH.value] == 3
        assert status[TaskType.TRANSCRIPTION.value] == 2
        assert status[TaskType.SCENE_DETECTION.value] == 0
