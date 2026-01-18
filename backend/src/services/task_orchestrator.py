"""Task orchestrator for managing video processing tasks."""

import logging
import uuid
from datetime import datetime

from ..domain.models import Task, Video
from ..repositories.interfaces import TaskRepository, VideoRepository
from .task_orchestration import (
    TaskDependencyManager,
    TaskPriority,
    TaskQueues,
    TaskType,
    is_video_ready_for_task_type,
)

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """Central orchestrator for managing video processing tasks."""

    def __init__(
        self,
        task_repository: TaskRepository,
        video_repository: VideoRepository,
        task_queues: TaskQueues | None = None,
        dependency_manager: TaskDependencyManager | None = None,
    ):
        self.task_repository = task_repository
        self.video_repository = video_repository
        self.task_queues = task_queues or TaskQueues()
        self.dependency_manager = dependency_manager or TaskDependencyManager()

    def create_tasks_for_video(self, video: Video) -> list[Task]:
        """Create all appropriate tasks for a video based on its current state."""
        created_tasks = []

        # Get ready task types for this video
        ready_task_types = []
        for task_type in TaskType:
            if is_video_ready_for_task_type(video, task_type):
                # Check if task already exists and is not failed
                existing_task = self._get_existing_task(video.video_id, task_type)
                if not existing_task or existing_task.is_failed():
                    ready_task_types.append(task_type)

        # Create tasks for ready types
        for task_type in ready_task_types:
            task = self._create_task(video.video_id, task_type)
            created_tasks.append(task)

            # Enqueue task with appropriate priority
            priority = self._get_task_priority(task_type)
            self.task_queues.enqueue(task, priority)

            logger.info(
                f"Created and enqueued {task_type.value} task for video "
                f"{video.video_id}"
            )

        return created_tasks

    def process_discovered_videos(self) -> int:
        """Process all discovered videos and create hash tasks."""
        discovered_videos = self.video_repository.find_by_status("discovered")
        total_created = 0

        logger.info(f"Found {len(discovered_videos)} discovered videos to process")

        for video in discovered_videos:
            logger.info(
                f"Processing discovered video: {video.video_id} - {video.filename}"
            )
            tasks = self.create_tasks_for_video(video)
            total_created += len(tasks)

            if tasks:
                logger.info(f"Created {len(tasks)} tasks for video {video.video_id}")
                for task in tasks:
                    logger.info(f"  - {task.task_type} task: {task.task_id}")
            else:
                logger.warning(f"No tasks created for video {video.video_id}")

        logger.info(
            f"Created {total_created} tasks for {len(discovered_videos)} "
            f"discovered videos"
        )
        return total_created

    def process_hashed_videos(self) -> int:
        """Process all hashed videos and create parallel tasks."""
        hashed_videos = self.video_repository.find_by_status("hashed")
        total_created = 0

        for video in hashed_videos:
            tasks = self.create_tasks_for_video(video)
            total_created += len(tasks)

            # Update video status to processing if tasks were created
            if tasks:
                video.mark_processing()
                self.video_repository.save(video)

        logger.info(
            f"Created {total_created} tasks for {len(hashed_videos)} hashed videos"
        )
        return total_created

    def handle_task_completion(self, task: Task) -> list[Task]:
        """Handle task completion and create dependent tasks if ready."""
        created_tasks = []

        # Mark task as completed in dependency manager
        task_type = TaskType(task.task_type)
        self.dependency_manager.mark_task_completed(task.video_id, task_type)

        # Update task status in database
        task.complete()
        self.task_repository.update(task)

        # Update video status if this was a hash task
        if task_type == TaskType.HASH:
            video = self.video_repository.find_by_id(task.video_id)
            if video:
                video.status = "hashed"
                self.video_repository.save(video)
                logger.info(f"Updated video {task.video_id} status to hashed")

        # Check if we can create dependent tasks
        video = self.video_repository.find_by_id(task.video_id)
        if video:
            new_tasks = self.create_tasks_for_video(video)
            created_tasks.extend(new_tasks)

        # Check if all tasks are complete for this video
        self._check_video_completion(task.video_id)

        logger.info(
            f"Completed {task_type.value} task for video {task.video_id}, "
            f"created {len(created_tasks)} new tasks"
        )
        return created_tasks

    def handle_task_failure(self, task: Task, error: str) -> None:
        """Handle task failure."""
        # Mark task as failed
        task.fail(error)
        self.task_repository.update(task)

        # Update video status if this was a critical failure
        task_type = TaskType(task.task_type)
        if task_type == TaskType.HASH:
            video = self.video_repository.find_by_id(task.video_id)
            if video:
                video.status = "failed"
                self.video_repository.save(video)

        logger.error(
            f"Task {task_type.value} failed for video {task.video_id}: {error}"
        )

    def get_next_task(self, task_type: TaskType) -> Task | None:
        """Get the next task of specified type from queue."""
        return self.task_queues.dequeue(task_type)

    def get_queue_status(self) -> dict:
        """Get status of all task queues."""
        return self.task_queues.get_all_queue_sizes()

    def retry_failed_tasks(self, video_id: str | None = None) -> int:
        """Retry failed tasks for a specific video or all videos."""
        if video_id:
            failed_tasks = self.task_repository.find_by_video_and_status(
                video_id, "failed"
            )
        else:
            failed_tasks = self.task_repository.find_by_status("failed")

        retried_count = 0
        for task in failed_tasks:
            # Reset task status and re-enqueue
            task.status = "pending"
            task.error = None
            task.started_at = None
            self.task_repository.update(task)

            # Re-enqueue task
            task_type = TaskType(task.task_type)
            priority = self._get_task_priority(task_type)
            self.task_queues.enqueue(task, priority)
            retried_count += 1

        logger.info(f"Retried {retried_count} failed tasks")
        return retried_count

    def _create_task(self, video_id: str, task_type: TaskType) -> Task:
        """Create a new task."""
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=video_id,
            task_type=task_type.value,
            status="pending",
            priority=self._get_task_priority(task_type),
            created_at=datetime.utcnow(),
        )

        # Save to database
        saved_task = self.task_repository.save(task)
        return saved_task

    def _get_existing_task(self, video_id: str, task_type: TaskType) -> Task | None:
        """Get existing task for video and type."""
        tasks = self.task_repository.find_by_video_and_type(video_id, task_type.value)
        return tasks[0] if tasks else None

    def _get_task_priority(self, task_type: TaskType) -> int:
        """Get priority for task type."""
        priority_map = {
            TaskType.HASH: TaskPriority.CRITICAL.value,
            TaskType.TRANSCRIPTION: TaskPriority.HIGH.value,
            TaskType.SCENE_DETECTION: TaskPriority.MEDIUM.value,
            TaskType.OBJECT_DETECTION: TaskPriority.MEDIUM.value,
            TaskType.FACE_DETECTION: TaskPriority.MEDIUM.value,
            TaskType.TOPIC_EXTRACTION: TaskPriority.LOW.value,
            TaskType.EMBEDDING_GENERATION: TaskPriority.HIGH.value,
            TaskType.THUMBNAIL_GENERATION: TaskPriority.LOW.value,
        }
        return priority_map.get(task_type, TaskPriority.MEDIUM.value)

    def _check_video_completion(self, video_id: str) -> None:
        """Check if all tasks are complete for a video and update status."""
        # Get all tasks for video
        all_tasks = self.task_repository.find_by_video_id(video_id)

        if not all_tasks:
            return

        # Check if all tasks are completed
        completed_tasks = [t for t in all_tasks if t.is_completed()]
        failed_tasks = [t for t in all_tasks if t.is_failed()]

        # If we have failed tasks, don't mark as complete
        if failed_tasks:
            return

        # Check if we have all expected task types completed
        completed_types = {TaskType(t.task_type) for t in completed_tasks}
        expected_types = {
            TaskType.HASH,
            TaskType.TRANSCRIPTION,
            TaskType.SCENE_DETECTION,
            TaskType.OBJECT_DETECTION,
            TaskType.FACE_DETECTION,
            TaskType.TOPIC_EXTRACTION,
            TaskType.EMBEDDING_GENERATION,
            TaskType.THUMBNAIL_GENERATION,
        }

        if expected_types.issubset(completed_types):
            # All tasks complete - mark video as completed
            video = self.video_repository.find_by_id(video_id)
            if video and video.status != "completed":
                video.mark_completed()
                self.video_repository.save(video)
                logger.info(f"Video {video_id} processing completed")
