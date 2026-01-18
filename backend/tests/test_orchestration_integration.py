"""Integration tests for task orchestration system."""

import uuid
from datetime import datetime
from unittest.mock import Mock

from src.domain.models import Task, Video
from src.services.processing_profiles import ProfileManager
from src.services.task_orchestration import TaskDependencyManager, TaskQueues, TaskType
from src.services.task_orchestrator import TaskOrchestrator
from src.services.worker_pool_manager import (
    WorkerPoolManager,
)


class TestTaskOrchestrationIntegration:
    """Integration tests for the complete task orchestration system."""

    def setup_method(self):
        """Set up integration test fixtures."""
        # Mock repositories
        self.task_repo = Mock()
        self.video_repo = Mock()

        # Create real orchestration components
        self.task_queues = TaskQueues()
        self.dependency_manager = TaskDependencyManager()
        self.orchestrator = TaskOrchestrator(
            self.task_repo, self.video_repo, self.task_queues, self.dependency_manager
        )
        self.worker_manager = WorkerPoolManager(self.orchestrator)
        self.profile_manager = ProfileManager()

    def test_end_to_end_video_processing_flow(self):
        """Test complete video processing flow from discovery to completion."""
        # Create a discovered video
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/video.mp4",
            filename="video.mp4",
            last_modified=datetime.utcnow(),
            status="discovered",
            file_hash=None,
            file_size=1000,
        )

        # Mock repository responses
        self.task_repo.save.side_effect = lambda task: task
        self.task_repo.find_by_video_and_type.return_value = []
        # Add this for completion check
        self.task_repo.find_by_video_id.return_value = []

        # Step 1: Create hash task for discovered video
        tasks = self.orchestrator.create_tasks_for_video(video)

        assert len(tasks) == 1
        assert tasks[0].task_type == TaskType.HASH.value
        assert self.task_queues.get_queue_size(TaskType.HASH) == 1

        # Step 2: Simulate hash task completion
        hash_task = tasks[0]
        video.status = "hashed"
        video.file_hash = "abc123"
        self.video_repo.find_by_id.return_value = video

        new_tasks = self.orchestrator.handle_task_completion(hash_task)

        # Should create parallel tasks
        expected_types = {
            TaskType.TRANSCRIPTION,
            TaskType.SCENE_DETECTION,
            TaskType.OBJECT_DETECTION,
            TaskType.FACE_DETECTION,
        }
        created_types = {TaskType(t.task_type) for t in new_tasks}

        assert expected_types.issubset(created_types)

        # Step 3: Simulate transcription completion to enable dependent tasks
        transcription_task = next(
            t for t in new_tasks if t.task_type == TaskType.TRANSCRIPTION.value
        )
        video.status = "processing"

        dependent_tasks = self.orchestrator.handle_task_completion(transcription_task)

        # Should create dependent tasks
        dependent_types = {TaskType(t.task_type) for t in dependent_tasks}
        expected_dependent = {TaskType.TOPIC_EXTRACTION, TaskType.EMBEDDING_GENERATION}

        assert expected_dependent.issubset(dependent_types)

    def test_profile_integration_with_worker_manager(self):
        """Test that processing profiles correctly configure worker pools."""
        # Get a processing profile
        profile = self.profile_manager.get_profile("balanced")

        # Configure worker manager with profile
        for task_type_str, worker_config in profile.worker_configs.items():
            self.worker_manager.add_worker_pool(worker_config)

        # Verify worker pools are configured correctly
        status = self.worker_manager.get_status()

        assert len(status) == len(profile.worker_configs)

        # Check specific configurations
        hash_status = status[TaskType.HASH.value]
        assert hash_status["worker_count"] == 4
        assert hash_status["resource_type"] == "cpu"

        transcription_status = status[TaskType.TRANSCRIPTION.value]
        assert transcription_status["worker_count"] == 2
        assert transcription_status["resource_type"] == "cpu"

    def test_task_dependency_chain_validation(self):
        """Test that task dependencies are properly enforced."""
        video_id = str(uuid.uuid4())

        # Initially, only hash task should be ready
        ready_tasks = self.dependency_manager.get_ready_task_types(video_id)
        assert TaskType.HASH in ready_tasks
        assert TaskType.TRANSCRIPTION not in ready_tasks
        assert TaskType.TOPIC_EXTRACTION not in ready_tasks

        # After hash completion, parallel tasks should be ready
        self.dependency_manager.mark_task_completed(video_id, TaskType.HASH)
        ready_tasks = self.dependency_manager.get_ready_task_types(video_id)

        parallel_tasks = {
            TaskType.TRANSCRIPTION,
            TaskType.SCENE_DETECTION,
            TaskType.OBJECT_DETECTION,
            TaskType.FACE_DETECTION,
        }
        assert parallel_tasks.issubset(set(ready_tasks))
        assert TaskType.TOPIC_EXTRACTION not in ready_tasks  # Still needs transcription

        # After transcription completion, dependent tasks should be ready
        self.dependency_manager.mark_task_completed(video_id, TaskType.TRANSCRIPTION)
        ready_tasks = self.dependency_manager.get_ready_task_types(video_id)

        assert TaskType.TOPIC_EXTRACTION in ready_tasks
        assert TaskType.EMBEDDING_GENERATION in ready_tasks

    def test_queue_priority_ordering_integration(self):
        """Test that tasks are queued and dequeued in correct priority order."""
        # Create tasks with different priorities
        hash_task = Task(
            task_id="hash_task",
            video_id=str(uuid.uuid4()),
            task_type=TaskType.HASH.value,
            status="pending",
        )

        transcription_task = Task(
            task_id="transcription_task",
            video_id=str(uuid.uuid4()),
            task_type=TaskType.TRANSCRIPTION.value,
            status="pending",
        )

        topic_task = Task(
            task_id="topic_task",
            video_id=str(uuid.uuid4()),
            task_type=TaskType.TOPIC_EXTRACTION.value,
            status="pending",
        )

        # Enqueue in reverse priority order
        self.task_queues.enqueue(topic_task, 4)  # Low priority
        self.task_queues.enqueue(transcription_task, 2)  # High priority
        self.task_queues.enqueue(hash_task, 1)  # Critical priority

        # Should dequeue in priority order (critical first)
        first_task = self.orchestrator.get_next_task(TaskType.HASH)
        assert first_task.task_id == "hash_task"

        second_task = self.orchestrator.get_next_task(TaskType.TRANSCRIPTION)
        assert second_task.task_id == "transcription_task"

        third_task = self.orchestrator.get_next_task(TaskType.TOPIC_EXTRACTION)
        assert third_task.task_id == "topic_task"

    def test_error_handling_integration(self):
        """Test error handling across orchestration components."""
        video = Video(
            video_id=str(uuid.uuid4()),
            file_path="/test/missing.mp4",
            filename="missing.mp4",
            last_modified=datetime.utcnow(),
            status="discovered",
            file_hash=None,
            file_size=1000,
        )

        # Mock task failure
        self.task_repo.save.side_effect = lambda task: task
        self.task_repo.find_by_video_and_type.return_value = []
        self.video_repo.find_by_id.return_value = video
        self.video_repo.update.return_value = video
        self.task_repo.update.return_value = None

        # Create and fail a hash task
        tasks = self.orchestrator.create_tasks_for_video(video)
        hash_task = tasks[0]

        # Simulate task failure
        error_msg = "File not found"
        self.orchestrator.handle_task_failure(hash_task, error_msg)

        # Task should be marked as failed
        assert hash_task.is_failed()
        assert hash_task.error == error_msg

        # Video status should be updated for critical hash failure
        self.video_repo.update.assert_called()

    def test_custom_profile_loading_and_usage(self):
        """Test loading custom profiles and using them with worker manager."""
        # Create a custom profile configuration
        custom_config = {
            "name": "test_profile",
            "description": "Test profile for integration testing",
            "workers": {
                TaskType.HASH.value: {"count": 8, "priority": 1, "resource": "cpu"},
                TaskType.TRANSCRIPTION.value: {
                    "count": 4,
                    "priority": 1,
                    "resource": "cpu",
                },
            },
            "task_settings": {
                "max_concurrent_videos": 10,
                "frame_sampling_interval": 15,
                "face_sampling_interval_seconds": 2.0,
                "transcription_model": "large-v3-turbo",
                "object_detection_model": "yolov8s.pt",
                "face_detection_model": "yolov8s-face.pt",
            },
        }

        # Load custom profile
        from src.services.processing_profiles import create_profile_from_config

        profile = create_profile_from_config(custom_config)
        self.profile_manager.add_profile(profile)

        # Verify profile is available
        profiles = self.profile_manager.list_profiles()
        assert "test_profile" in profiles

        # Configure worker manager with custom profile
        retrieved_profile = self.profile_manager.get_profile("test_profile")

        for task_type_str, worker_config in retrieved_profile.worker_configs.items():
            self.worker_manager.add_worker_pool(worker_config)

        # Verify custom configuration is applied
        status = self.worker_manager.get_status()
        assert status[TaskType.HASH.value]["worker_count"] == 8
        assert status[TaskType.TRANSCRIPTION.value]["worker_count"] == 4
