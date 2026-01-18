"""Tests for task orchestration system."""

import uuid
from datetime import datetime

from src.domain.models import Task, Video
from src.services.task_orchestration import (
    TaskDependencyManager,
    TaskPriority,
    TaskQueues,
    TaskType,
    is_video_ready_for_task_type,
)


class TestTaskQueues:
    """Test task queue functionality."""

    def test_enqueue_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        queues = TaskQueues()

        # Create a test task
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        # Enqueue task
        queues.enqueue(task, TaskPriority.HIGH.value)

        # Check queue size
        assert queues.get_queue_size(TaskType.HASH) == 1

        # Dequeue task
        dequeued_task = queues.dequeue(TaskType.HASH)
        assert dequeued_task is not None
        assert dequeued_task.task_id == task.task_id

        # Queue should be empty now
        assert queues.get_queue_size(TaskType.HASH) == 0

    def test_priority_ordering(self):
        """Test that tasks are dequeued in priority order."""
        queues = TaskQueues()

        # Create tasks with different priorities
        low_task = Task(
            task_id="low",
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        high_task = Task(
            task_id="high",
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        # Enqueue in reverse priority order
        queues.enqueue(low_task, TaskPriority.LOW.value)
        queues.enqueue(high_task, TaskPriority.HIGH.value)

        # High priority should come out first
        first_task = queues.dequeue(TaskType.HASH)
        assert first_task.task_id == "high"

        second_task = queues.dequeue(TaskType.HASH)
        assert second_task.task_id == "low"


class TestTaskDependencyManager:
    """Test task dependency management."""

    def test_hash_task_ready_immediately(self):
        """Test that hash task has no dependencies."""
        manager = TaskDependencyManager()
        video_id = str(uuid.uuid4())

        # Hash task should be ready immediately
        assert manager.is_task_ready(video_id, TaskType.HASH)

    def test_parallel_tasks_depend_on_hash(self):
        """Test that parallel tasks depend on hash completion."""
        manager = TaskDependencyManager()
        video_id = str(uuid.uuid4())

        # Parallel tasks should not be ready initially
        assert not manager.is_task_ready(video_id, TaskType.TRANSCRIPTION)
        assert not manager.is_task_ready(video_id, TaskType.SCENE_DETECTION)

        # Mark hash as completed
        manager.mark_task_completed(video_id, TaskType.HASH)

        # Now parallel tasks should be ready
        assert manager.is_task_ready(video_id, TaskType.TRANSCRIPTION)
        assert manager.is_task_ready(video_id, TaskType.SCENE_DETECTION)
        assert manager.is_task_ready(video_id, TaskType.OBJECT_DETECTION)
        assert manager.is_task_ready(video_id, TaskType.FACE_DETECTION)

    def test_dependent_tasks_need_transcription(self):
        """Test that topic/embedding tasks depend on transcription."""
        manager = TaskDependencyManager()
        video_id = str(uuid.uuid4())

        # Mark hash as completed
        manager.mark_task_completed(video_id, TaskType.HASH)

        # Topic extraction should not be ready yet
        assert not manager.is_task_ready(video_id, TaskType.TOPIC_EXTRACTION)
        assert not manager.is_task_ready(video_id, TaskType.EMBEDDING_GENERATION)

        # Mark transcription as completed
        manager.mark_task_completed(video_id, TaskType.TRANSCRIPTION)

        # Now dependent tasks should be ready
        assert manager.is_task_ready(video_id, TaskType.TOPIC_EXTRACTION)
        assert manager.is_task_ready(video_id, TaskType.EMBEDDING_GENERATION)

    def test_get_ready_task_types(self):
        """Test getting all ready task types for a video."""
        manager = TaskDependencyManager()
        video_id = str(uuid.uuid4())

        # Initially only hash should be ready
        ready_tasks = manager.get_ready_task_types(video_id)
        assert TaskType.HASH in ready_tasks
        assert len(ready_tasks) == 1

        # Mark hash as completed
        manager.mark_task_completed(video_id, TaskType.HASH)

        # Now parallel tasks should be ready
        ready_tasks = manager.get_ready_task_types(video_id)
        expected_parallel = {
            TaskType.TRANSCRIPTION,
            TaskType.SCENE_DETECTION,
            TaskType.OBJECT_DETECTION,
            TaskType.FACE_DETECTION,
        }
        assert expected_parallel.issubset(set(ready_tasks))


class TestVideoReadiness:
    """Test video readiness for different task types."""

    def test_hash_task_readiness(self):
        """Test hash task readiness based on video status."""
        # Video discovered, no hash - ready for hash
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="discovered",
            file_hash=None,
            file_size=1000,
        )

        assert is_video_ready_for_task_type(video, TaskType.HASH)

        # Video with hash - not ready for hash
        video.file_hash = "abc123"
        assert not is_video_ready_for_task_type(video, TaskType.HASH)

    def test_parallel_task_readiness(self):
        """Test parallel task readiness based on video status."""
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="discovered",
            file_hash=None,
            file_size=1000,
        )

        # Not ready when discovered without hash
        assert not is_video_ready_for_task_type(video, TaskType.TRANSCRIPTION)

        # Ready when hashed
        video.status = "hashed"
        video.file_hash = "abc123"
        assert is_video_ready_for_task_type(video, TaskType.TRANSCRIPTION)
        assert is_video_ready_for_task_type(video, TaskType.SCENE_DETECTION)

    def test_dependent_task_readiness(self):
        """Test dependent task readiness based on video status."""
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/test.mp4",
            filename="test.mp4",
            last_modified=datetime.utcnow(),
            status="hashed",
            file_hash="abc123",
            file_size=1000,
        )

        # Not ready when just hashed
        assert not is_video_ready_for_task_type(video, TaskType.TOPIC_EXTRACTION)

        # Ready when processing or completed
        video.status = "processing"
        assert is_video_ready_for_task_type(video, TaskType.TOPIC_EXTRACTION)

        video.status = "completed"
        assert is_video_ready_for_task_type(video, TaskType.TOPIC_EXTRACTION)
