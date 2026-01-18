"""Task orchestration system for video processing."""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import PriorityQueue

from ..domain.models import Task, Video

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of all task types in processing order."""

    HASH = "hash"
    TRANSCRIPTION = "transcription"
    SCENE_DETECTION = "scene_detection"
    OBJECT_DETECTION = "object_detection"
    FACE_DETECTION = "face_detection"
    TOPIC_EXTRACTION = "topic_extraction"
    EMBEDDING_GENERATION = "embedding_generation"
    THUMBNAIL_GENERATION = "thumbnail_generation"


class TaskStatus(Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    """Task priority levels."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class TaskQueueItem:
    """Item in task queue with priority support."""

    priority: int
    created_at: datetime
    task: Task

    def __lt__(self, other):
        """Compare for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class TaskQueues:
    """Manages separate queues for each task type."""

    def __init__(self):
        self._queues: dict[TaskType, PriorityQueue] = {}
        self._locks: dict[TaskType, threading.Lock] = {}

        # Initialize queues and locks for each task type
        for task_type in TaskType:
            self._queues[task_type] = PriorityQueue()
            self._locks[task_type] = threading.Lock()

    def enqueue(self, task: Task, priority: int = TaskPriority.MEDIUM.value) -> None:
        """Add task to appropriate queue."""
        task_type = TaskType(task.task_type)
        queue_item = TaskQueueItem(
            priority=priority, created_at=datetime.utcnow(), task=task
        )

        with self._locks[task_type]:
            self._queues[task_type].put(queue_item)
            logger.info(f"Enqueued {task_type.value} task for video {task.video_id}")

    def dequeue(self, task_type: TaskType) -> Task | None:
        """Get next task from specified queue."""
        with self._locks[task_type]:
            if not self._queues[task_type].empty():
                queue_item = self._queues[task_type].get()
                logger.info(
                    f"Dequeued {task_type.value} task for video "
                    f"{queue_item.task.video_id}"
                )
                return queue_item.task
        return None

    def get_queue_size(self, task_type: TaskType) -> int:
        """Get current size of specified queue."""
        with self._locks[task_type]:
            return self._queues[task_type].qsize()

    def get_all_queue_sizes(self) -> dict[str, int]:
        """Get sizes of all queues."""
        return {
            task_type.value: self.get_queue_size(task_type) for task_type in TaskType
        }


class TaskDependencyManager:
    """Manages task dependencies and determines task readiness."""

    # Define task dependencies
    TASK_DEPENDENCIES = {
        TaskType.HASH: [],  # No dependencies
        TaskType.TRANSCRIPTION: [TaskType.HASH],
        TaskType.SCENE_DETECTION: [TaskType.HASH],
        TaskType.OBJECT_DETECTION: [TaskType.HASH],
        TaskType.FACE_DETECTION: [TaskType.HASH],
        TaskType.TOPIC_EXTRACTION: [TaskType.HASH, TaskType.TRANSCRIPTION],
        TaskType.EMBEDDING_GENERATION: [TaskType.HASH, TaskType.TRANSCRIPTION],
        TaskType.THUMBNAIL_GENERATION: [TaskType.HASH, TaskType.SCENE_DETECTION],
    }

    def __init__(self):
        self._completed_tasks: dict[
            str, set[TaskType]
        ] = {}  # video_id -> completed task types
        self._lock = threading.Lock()

    def mark_task_completed(self, video_id: str, task_type: TaskType) -> None:
        """Mark a task as completed for a video."""
        with self._lock:
            if video_id not in self._completed_tasks:
                self._completed_tasks[video_id] = set()
            self._completed_tasks[video_id].add(task_type)
            logger.info(f"Marked {task_type.value} as completed for video {video_id}")

    def is_task_ready(self, video_id: str, task_type: TaskType) -> bool:
        """Check if all dependencies for a task are completed."""
        with self._lock:
            completed = self._completed_tasks.get(video_id, set())
            dependencies = self.TASK_DEPENDENCIES[task_type]

            # Check if all dependencies are completed
            for dep in dependencies:
                if dep not in completed:
                    return False

            return True

    def get_ready_task_types(self, video_id: str) -> list[TaskType]:
        """Get all task types that are ready to run for a video."""
        ready_tasks = []
        for task_type in TaskType:
            if self.is_task_ready(video_id, task_type):
                # Also check if this task hasn't been completed yet
                with self._lock:
                    completed = self._completed_tasks.get(video_id, set())
                    if task_type not in completed:
                        ready_tasks.append(task_type)

        return ready_tasks

    def get_completed_tasks(self, video_id: str) -> set[TaskType]:
        """Get all completed tasks for a video."""
        with self._lock:
            return self._completed_tasks.get(video_id, set()).copy()


def is_video_ready_for_task_type(video: Video, task_type: TaskType) -> bool:
    """Check if video is ready for a specific task type based on status and hash."""
    if task_type == TaskType.HASH:
        # Hash task is ready if video is discovered and has no hash
        return video.status == "discovered" and video.file_hash is None

    elif task_type in [
        TaskType.TRANSCRIPTION,
        TaskType.SCENE_DETECTION,
        TaskType.OBJECT_DETECTION,
        TaskType.FACE_DETECTION,
    ]:
        # Parallel tasks are ready if video is hashed
        return video.status == "hashed" and video.file_hash is not None

    elif task_type in [
        TaskType.TOPIC_EXTRACTION,
        TaskType.EMBEDDING_GENERATION,
        TaskType.THUMBNAIL_GENERATION,
    ]:
        # Dependent tasks are ready if video is processing or completed
        return (
            video.status in ["processing", "completed"] and video.file_hash is not None
        )

    return False
