"""Worker pool management for video processing tasks."""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..domain.models import Task
from .task_orchestration import TaskType
from .task_orchestrator import TaskOrchestrator

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource types for workers."""

    CPU = "cpu"
    GPU = "gpu"
    IO = "io"


@dataclass
class WorkerConfig:
    """Configuration for a worker pool."""

    task_type: TaskType
    worker_count: int
    resource_type: ResourceType
    priority: int = 3  # 1=critical, 2=high, 3=medium, 4=low


class TaskWorker:
    """Base class for task workers."""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.logger = logging.getLogger(f"worker.{task_type.value}")

    def execute_task(self, task: Task) -> dict:
        """Execute a task and return result."""
        self.logger.info(f"Starting {self.task_type.value} task {task.task_id}")

        try:
            # Mark task as running
            task.start()

            # Execute the actual work
            result = self._do_work(task)

            # Mark task as completed
            task.complete()

            self.logger.info(f"Completed {self.task_type.value} task {task.task_id}")
            return {"status": "success", "result": result}

        except Exception as e:
            error_msg = str(e)
            task.fail(error_msg)
            self.logger.error(
                f"Failed {self.task_type.value} task {task.task_id}: {error_msg}"
            )
            return {"status": "error", "error": error_msg}

    def _do_work(self, task: Task) -> Any:
        """Override this method to implement actual work."""
        # Simulate work for now
        time.sleep(0.1)
        return f"Processed {self.task_type.value} for video {task.video_id}"


class HashWorker(TaskWorker):
    """Worker for hash calculation tasks."""

    def __init__(self):
        super().__init__(TaskType.HASH)

    def _do_work(self, task: Task) -> str:
        """Calculate file hash."""
        # TODO: Implement actual hash calculation
        import hashlib

        # Simulate hash calculation
        time.sleep(0.05)  # Fast operation
        return hashlib.sha256(f"fake_content_{task.video_id}".encode()).hexdigest()


class TranscriptionWorker(TaskWorker):
    """Worker for transcription tasks."""

    def __init__(self):
        super().__init__(TaskType.TRANSCRIPTION)

    def _do_work(self, task: Task) -> dict:
        """Perform transcription."""
        # TODO: Implement actual transcription with Whisper
        time.sleep(2.0)  # Simulate longer processing
        return {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Sample transcription segment"}
            ]
        }


class SceneDetectionWorker(TaskWorker):
    """Worker for scene detection tasks."""

    def __init__(self):
        super().__init__(TaskType.SCENE_DETECTION)

    def _do_work(self, task: Task) -> dict:
        """Perform scene detection."""
        # TODO: Implement actual scene detection
        time.sleep(1.0)  # Simulate processing
        return {"scenes": [{"start": 0.0, "end": 10.0, "scene_id": 1}]}


class ObjectDetectionWorker(TaskWorker):
    """Worker for object detection tasks."""

    def __init__(self):
        super().__init__(TaskType.OBJECT_DETECTION)

    def _do_work(self, task: Task) -> dict:
        """Perform object detection."""
        # TODO: Implement actual object detection with YOLO
        time.sleep(1.5)  # Simulate GPU processing
        return {
            "objects": [
                {"label": "person", "confidence": 0.95, "timestamps": [1.0, 2.0]}
            ]
        }


class WorkerPool:
    """Manages a pool of workers for a specific task type."""

    def __init__(self, config: WorkerConfig, orchestrator: TaskOrchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.is_running = False
        self.workers = []
        self.executor = None
        self._stop_event = threading.Event()

        # Create worker instances
        self.worker_factory = self._get_worker_factory()

    def start(self) -> None:
        """Start the worker pool."""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        # Create appropriate executor based on resource type
        if self.config.resource_type == ResourceType.CPU:
            self.executor = ProcessPoolExecutor(max_workers=self.config.worker_count)
        else:
            # Use thread pool for GPU/IO tasks to avoid multiprocessing overhead
            self.executor = ThreadPoolExecutor(max_workers=self.config.worker_count)

        # Start worker threads
        for i in range(self.config.worker_count):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self.config.task_type.value}-worker-{i}",
                daemon=True,
            )
            worker_thread.start()
            self.workers.append(worker_thread)

        logger.info(
            f"Started {self.config.worker_count} workers for "
            f"{self.config.task_type.value}"
        )

    def stop(self) -> None:
        """Stop the worker pool."""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)

        # Wait for worker threads to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()
        logger.info(f"Stopped worker pool for {self.config.task_type.value}")

    def _worker_loop(self) -> None:
        """Main loop for worker threads."""
        worker = self.worker_factory()

        while self.is_running and not self._stop_event.is_set():
            try:
                # Get next task from orchestrator
                task = self.orchestrator.get_next_task(self.config.task_type)

                if task is None:
                    # No tasks available, wait a bit
                    time.sleep(0.1)
                    continue

                # Submit task to executor
                future = self.executor.submit(worker.execute_task, task)

                # Wait for completion and handle result
                try:
                    result = future.result(timeout=300)  # 5 minute timeout

                    if result["status"] == "success":
                        self.orchestrator.handle_task_completion(task)
                    else:
                        self.orchestrator.handle_task_failure(task, result["error"])

                except Exception as e:
                    error_msg = f"Worker execution failed: {str(e)}"
                    self.orchestrator.handle_task_failure(task, error_msg)

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1.0)  # Prevent tight error loops

    def _get_worker_factory(self) -> Callable[[], TaskWorker]:
        """Get worker factory for task type."""
        factories = {
            TaskType.HASH: HashWorker,
            TaskType.TRANSCRIPTION: TranscriptionWorker,
            TaskType.SCENE_DETECTION: SceneDetectionWorker,
            TaskType.OBJECT_DETECTION: ObjectDetectionWorker,
            TaskType.FACE_DETECTION: TaskWorker,  # Generic for now
            TaskType.TOPIC_EXTRACTION: TaskWorker,  # Generic for now
            TaskType.EMBEDDING_GENERATION: TaskWorker,  # Generic for now
            TaskType.THUMBNAIL_GENERATION: TaskWorker,  # Generic for now
        }
        return factories.get(self.config.task_type, TaskWorker)


class WorkerPoolManager:
    """Manages multiple worker pools."""

    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.pools: dict[TaskType, WorkerPool] = {}
        self.is_running = False

    def add_worker_pool(self, config: WorkerConfig) -> None:
        """Add a worker pool for a task type."""
        if config.task_type in self.pools:
            raise ValueError(f"Worker pool for {config.task_type.value} already exists")

        pool = WorkerPool(config, self.orchestrator)
        self.pools[config.task_type] = pool

        # Start pool if manager is running
        if self.is_running:
            pool.start()

        logger.info(f"Added worker pool for {config.task_type.value}")

    def start_all(self) -> None:
        """Start all worker pools."""
        if self.is_running:
            return

        self.is_running = True

        for pool in self.pools.values():
            pool.start()

        logger.info(f"Started {len(self.pools)} worker pools")

    def stop_all(self) -> None:
        """Stop all worker pools."""
        if not self.is_running:
            return

        self.is_running = False

        for pool in self.pools.values():
            pool.stop()

        logger.info("Stopped all worker pools")

    def get_status(self) -> dict[str, dict]:
        """Get status of all worker pools."""
        status = {}
        for task_type, pool in self.pools.items():
            status[task_type.value] = {
                "worker_count": pool.config.worker_count,
                "resource_type": pool.config.resource_type.value,
                "is_running": pool.is_running,
                "active_workers": len([w for w in pool.workers if w.is_alive()]),
            }
        return status

    def create_default_pools(self) -> None:
        """Create default worker pools with balanced configuration."""
        default_configs = [
            WorkerConfig(TaskType.HASH, 4, ResourceType.CPU, 1),
            WorkerConfig(TaskType.TRANSCRIPTION, 2, ResourceType.CPU, 2),
            WorkerConfig(TaskType.SCENE_DETECTION, 2, ResourceType.CPU, 3),
            WorkerConfig(TaskType.OBJECT_DETECTION, 2, ResourceType.GPU, 3),
            WorkerConfig(TaskType.FACE_DETECTION, 2, ResourceType.GPU, 3),
            WorkerConfig(TaskType.TOPIC_EXTRACTION, 1, ResourceType.CPU, 4),
            WorkerConfig(TaskType.EMBEDDING_GENERATION, 2, ResourceType.CPU, 2),
            WorkerConfig(TaskType.THUMBNAIL_GENERATION, 1, ResourceType.CPU, 4),
        ]

        for config in default_configs:
            self.add_worker_pool(config)
