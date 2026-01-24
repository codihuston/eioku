"""Worker pool management for video processing tasks."""

import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..domain.models import Task
from ..utils.print_logger import get_logger
from .task_orchestration import TaskType
from .task_orchestrator import TaskOrchestrator

if TYPE_CHECKING:
    from ..repositories.video_repository import SqlVideoRepository as VideoRepository
    from .object_detection_task_handler import ObjectDetectionTaskHandler

logger = get_logger(__name__)


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
    priority: int


class TaskWorker:
    """Base class for task workers."""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.logger = get_logger(f"worker.{task_type.value}")

    def execute_task(self, task: Task) -> dict:
        """Execute a task and return result."""
        self.logger.info(
            f"Starting {self.task_type.value} task {task.task_id} "
            f"for video {task.video_id}"
        )

        try:
            # Task is already marked as running by atomic dequeue

            # Execute the actual work
            result = self._do_work(task)

            # Mark task as completed
            task.complete()

            self.logger.info(
                f"Completed {self.task_type.value} task {task.task_id} "
                f"for video {task.video_id}"
            )
            return {"status": "success", "result": result}

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            task.fail(error_msg)
            self.logger.error(
                f"Failed {self.task_type.value} task {task.task_id} "
                f"for video {task.video_id}: {error_msg}\n{traceback.format_exc()}"
            )
            return {"status": "error", "error": error_msg}

    def _do_work(self, task: Task) -> Any:
        """Override this method to implement actual work."""
        # Simulate work for now
        time.sleep(0.1)
        return f"Processed {self.task_type.value} for video {task.video_id}"


class HashWorker(TaskWorker):
    """Worker for hash calculation tasks."""

    def __init__(self, hash_service=None, video_repository=None):
        super().__init__(TaskType.HASH)
        from .file_hash_service import FileHashService

        self.hash_service = hash_service or FileHashService()
        self.video_repository = video_repository

    def _do_work(self, task: Task) -> str:
        """Calculate file hash."""
        # Get video from repository to get the file path
        if not self.video_repository:
            raise RuntimeError("HashWorker requires a video_repository")

        video = self.video_repository.find_by_id(task.video_id)
        if not video:
            raise RuntimeError(f"Video not found: {task.video_id}")

        try:
            hash_value = self.hash_service.calculate_hash(video.file_path)
            logger.info(f"Calculated hash for {task.video_id}: {hash_value}")

            # Update video with hash
            video.file_hash = hash_value
            self.video_repository.save(video)

            return hash_value
        except Exception as e:
            logger.error(f"Hash calculation failed for {task.video_id}: {e}")
            raise


class ObjectDetectionWorker(TaskWorker):
    """Worker for object detection tasks."""

    def __init__(
        self,
        detection_handler: "ObjectDetectionTaskHandler",
        video_repository: "VideoRepository",
        model_profile: str = "balanced",
    ):
        super().__init__(TaskType.OBJECT_DETECTION)
        self.detection_handler = detection_handler
        self.video_repository = video_repository
        self.model_profile = model_profile

    def _do_work(self, task: Task) -> dict:
        """Perform object detection."""
        # Get video
        video = self.video_repository.find_by_id(task.video_id)
        if not video:
            raise ValueError(f"Video not found: {task.video_id}")

        # Process object detection with model_profile
        success = self.detection_handler.process_object_detection_task(
            task, video, model_profile=self.model_profile
        )

        if not success:
            raise Exception("Object detection processing failed")

        # Return count from processed artifacts (avoid querying after batch_create)
        return {
            "objects_detected": "processed",
            "message": "Object detection completed successfully",
        }


class FaceDetectionWorker(TaskWorker):
    """Worker for face detection tasks."""

    def __init__(
        self,
        detection_handler,
        video_repository: "VideoRepository",
        model_profile: str = "balanced",
    ):
        super().__init__(TaskType.FACE_DETECTION)
        self.detection_handler = detection_handler
        self.video_repository = video_repository
        self.model_profile = model_profile

    def _do_work(self, task: Task) -> dict:
        """Perform face detection."""
        # Get video
        video = self.video_repository.find_by_id(task.video_id)
        if not video:
            raise ValueError(f"Video not found: {task.video_id}")

        # Process face detection with model_profile
        success = self.detection_handler.process_face_detection_task(
            task, video, model_profile=self.model_profile
        )

        if not success:
            raise Exception("Face detection processing failed")

        # Return success message (avoid querying after batch_create)
        return {
            "message": "Face detection completed successfully",
        }


class OcrWorker(TaskWorker):
    """Worker for OCR text detection tasks."""

    def __init__(
        self,
        ocr_handler,
        video_repository: "VideoRepository",
        model_profile: str = "balanced",
    ):
        super().__init__(TaskType.OCR)
        self.ocr_handler = ocr_handler
        self.video_repository = video_repository
        self.model_profile = model_profile

    def _do_work(self, task: Task) -> dict:
        """Perform OCR text detection."""
        # Get video
        video = self.video_repository.find_by_id(task.video_id)
        if not video:
            raise ValueError(f"Video not found: {task.video_id}")

        # Process OCR with model_profile
        success = self.ocr_handler.process_ocr_task(
            task, video, model_profile=self.model_profile
        )

        if not success:
            raise Exception("OCR processing failed")

        # Return success message (avoid querying after batch_create)
        return {
            "message": "OCR completed successfully",
        }


class PlaceDetectionWorker(TaskWorker):
    """Worker for place classification tasks."""

    def __init__(
        self,
        detection_handler,
        video_repository: "VideoRepository",
        model_profile: str = "balanced",
    ):
        super().__init__(TaskType.PLACE_DETECTION)
        self.detection_handler = detection_handler
        self.video_repository = video_repository
        self.model_profile = model_profile

    def _do_work(self, task: Task) -> dict:
        """Perform place detection."""
        # Get video
        video = self.video_repository.find_by_id(task.video_id)
        if not video:
            raise ValueError(f"Video not found: {task.video_id}")

        # Process place detection with model_profile
        success = self.detection_handler.process_place_detection_task(
            task, video, model_profile=self.model_profile
        )

        if not success:
            raise Exception("Place detection processing failed")

        # Return success message (avoid querying after batch_create)
        return {
            "message": "Place detection completed successfully",
        }


class TranscriptionWorker(TaskWorker):
    """Worker for transcription tasks."""

    def __init__(
        self,
        transcription_handler,
        video_repository: "VideoRepository",
        model_profile: str = "balanced",
    ):
        super().__init__(TaskType.TRANSCRIPTION)
        self.transcription_handler = transcription_handler
        self.video_repository = video_repository
        self.model_profile = model_profile

    def _do_work(self, task: Task) -> dict:
        """Perform transcription."""
        # Get video
        video = self.video_repository.find_by_id(task.video_id)
        if not video:
            raise ValueError(f"Video not found: {task.video_id}")

        # Generate run_id for this transcription
        import uuid

        run_id = str(uuid.uuid4())

        # Process transcription with model_profile
        success = self.transcription_handler.process_transcription_task(
            task, video, run_id=run_id, model_profile=self.model_profile
        )

        if not success:
            raise Exception("Transcription processing failed")

        # Return success message (avoid querying after batch_create)
        return {
            "message": "Transcription completed successfully",
            "run_id": run_id,
        }


class SceneDetectionWorker(TaskWorker):
    """Worker for scene detection tasks."""

    def __init__(
        self,
        scene_service,
        artifact_repository,
        video_repository: "VideoRepository",
        model_profile: str = "balanced",
    ):
        super().__init__(TaskType.SCENE_DETECTION)
        self.scene_service = scene_service
        self.artifact_repository = artifact_repository
        self.video_repository = video_repository
        self.model_profile = model_profile

    def _do_work(self, task: Task) -> dict:
        """Perform scene detection."""
        # Get video
        video = self.video_repository.find_by_id(task.video_id)
        if not video:
            raise ValueError(f"Video not found: {task.video_id}")

        # Generate run_id for this scene detection
        import uuid

        run_id = str(uuid.uuid4())

        # Detect scenes and create artifacts
        artifacts = self.scene_service.detect_scenes(
            video.file_path, video.video_id, run_id, model_profile=self.model_profile
        )

        # Save artifacts to repository using batch create
        self.artifact_repository.batch_create(artifacts)

        # Get scene info for response
        scene_info = self.scene_service.get_scene_info(artifacts)

        return {
            "scenes_detected": scene_info["scene_count"],
            "run_id": run_id,
        }


class WorkerPool:
    """Manages a pool of workers for a specific task type."""

    def __init__(
        self,
        config: WorkerConfig,
        orchestrator: TaskOrchestrator,
        task_settings: dict | None = None,
        profile_name: str = "balanced",
    ):
        self.config = config
        self.orchestrator = orchestrator
        self.task_settings = task_settings or {}
        self.profile_name = profile_name  # Store profile name for model_profile
        self.is_running = False
        self.workers = []
        self.executor = None
        self._stop_event = threading.Event()

        # Get task timeout from config (default to 30 minutes = 1800 seconds)
        self.task_timeout = self.task_settings.get("task_timeout_seconds", 1800)
        if "task_timeout_seconds" not in self.task_settings:
            logger.info(
                f"Task timeout not configured for {config.task_type.value}, "
                f"using default: {self.task_timeout} seconds (30 minutes)"
            )

        # Create worker instances
        self.worker_factory = self._get_worker_factory()

    def start(self) -> None:
        """Start the worker pool."""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        # Use thread pool for all tasks to avoid pickling issues with database sessions
        self.executor = ThreadPoolExecutor(max_workers=self.config.worker_count)

        # Start worker threads with jitter to prevent simultaneous polling
        import random

        for i in range(self.config.worker_count):
            # Add random jitter (0-5 seconds) to stagger worker startup
            jitter = random.uniform(0, 5)

            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self.config.task_type.value}-worker-{i}",
                args=(jitter,),
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

    def _worker_loop(self, initial_jitter: float = 0) -> None:
        """Main loop for worker threads."""
        from ..database.connection import get_db
        from ..repositories.task_repository import SQLAlchemyTaskRepository
        from ..repositories.video_repository import SqlVideoRepository

        # Apply initial jitter to stagger worker startup
        if initial_jitter > 0:
            time.sleep(initial_jitter)
            logger.info(
                f"Worker loop started for {self.config.task_type.value} "
                f"(jitter: {initial_jitter:.2f}s)"
            )
        else:
            logger.info(f"Worker loop started for {self.config.task_type.value}")

        try:
            while self.is_running and not self._stop_event.is_set():
                # Create a new session for each task to ensure isolation
                session = None
                try:
                    # Get next task from database atomically
                    logger.debug(f"Checking for {self.config.task_type.value} tasks...")

                    # Create session for task dequeue
                    session = next(get_db())
                    task_repo = SQLAlchemyTaskRepository(session)

                    # Use atomic dequeue to prevent race conditions
                    task = task_repo.atomic_dequeue_pending_task(
                        self.config.task_type.value
                    )

                    if task is None:
                        # No tasks available, close session and wait
                        session.close()
                        session = None
                        time.sleep(30.0)
                        continue

                    logger.info(
                        f"Found task {task.task_id} for {self.config.task_type.value} "
                        f"(video: {task.video_id})"
                    )

                    # Create worker WITHOUT session - it will create its own
                    # Sessions cannot be shared across threads
                    worker = self.worker_factory()

                    # Submit task to executor
                    future = self.executor.submit(worker.execute_task, task)

                    # Wait for completion and handle result
                    try:
                        result = future.result(timeout=self.task_timeout)

                        if result["status"] == "success":
                            # Update task status in database
                            task.complete()
                            task_repo.update(task)

                            # Update video status if this was a hash task
                            if self.config.task_type == TaskType.HASH:
                                video_repo = SqlVideoRepository(session)
                                video = video_repo.find_by_id(task.video_id)
                                if video:
                                    video.status = "hashed"
                                    video_repo.save(video)
                                    logger.info(
                                        f"Updated video {task.video_id} to hashed"
                                    )

                                    # Create next tasks for the video using orchestrator
                                    new_tasks = (
                                        self.orchestrator.create_tasks_for_video(video)
                                    )
                                    if new_tasks:
                                        logger.info(
                                            f"Created {len(new_tasks)} new tasks"
                                        )
                                        for new_task in new_tasks:
                                            logger.info(
                                                f"   - {new_task.task_type} task"
                                            )

                            task_type = self.config.task_type.value
                            logger.info(
                                f"Completed {task_type} task {task.task_id} "
                                f"for video {task.video_id}"
                            )
                        else:
                            # Update task as failed
                            task.fail(result["error"])
                            task_repo.update(task)
                            logger.error(
                                f"Task {task.task_id} for video {task.video_id} "
                                f"failed: {result['error']}"
                            )

                    except Exception as e:
                        error_msg = f"Worker execution failed: {str(e)}"
                        # Rollback session before updating task
                        try:
                            session.rollback()
                        except Exception:
                            pass
                        task.fail(error_msg)
                        task_repo.update(task)
                        logger.error(
                            f"Task {task.task_id} for video {task.video_id} "
                            f"timed out: {error_msg}"
                        )
                    except Exception as e:
                        error_msg = (
                            f"Worker execution failed: {type(e).__name__}: {str(e)}"
                        )
                        error_details = traceback.format_exc()
                        # Rollback session before updating task
                        try:
                            session.rollback()
                        except Exception:
                            pass
                        task.fail(error_msg)
                        task_repo.update(task)
                        logger.error(
                            f"Task {task.task_id} for video {task.video_id} "
                            f"failed: {error_msg}\n{error_details}"
                        )

                except Exception as e:
                    logger.error(f"Worker loop error: {e}")
                    # Rollback the session to recover from transaction errors
                    if session:
                        try:
                            session.rollback()
                        except Exception as rollback_error:
                            logger.error(
                                f"Failed to rollback session: {rollback_error}"
                            )
                    time.sleep(1.0)  # Prevent tight error loops
                finally:
                    # Always close session after each task
                    if session:
                        try:
                            session.close()
                        except Exception as close_error:
                            logger.error(f"Failed to close session: {close_error}")
                        session = None

        except Exception as e:
            logger.error(f"Fatal worker loop error: {e}")
        finally:
            logger.info(f"Worker loop exiting for {self.config.task_type.value}")

    def _get_worker_factory(self) -> Callable[..., TaskWorker]:
        """Get worker factory for task type.

        Workers create their own database sessions to ensure thread safety.
        Sessions cannot be shared across threads.
        """
        if self.config.task_type == TaskType.HASH:
            from ..database.connection import get_db
            from ..repositories.video_repository import SqlVideoRepository
            from .file_hash_service import FileHashService

            def create_hash_worker():
                # Create session in worker thread
                session = next(get_db())
                video_repo = SqlVideoRepository(session)
                hash_service = FileHashService()
                return HashWorker(
                    hash_service=hash_service, video_repository=video_repo
                )

            return create_hash_worker
        elif self.config.task_type == TaskType.TRANSCRIPTION:
            from ..database.connection import get_db
            from ..domain.schema_registry import SchemaRegistry
            from ..repositories.artifact_repository import SqlArtifactRepository
            from ..repositories.video_repository import SqlVideoRepository
            from ..services.projection_sync_service import ProjectionSyncService
            from .transcription_task_handler import TranscriptionTaskHandler

            # Get settings from task_settings
            model_name = self.task_settings.get("transcription_model", "base")

            # Create transcription worker with dependencies
            def create_transcription_worker():
                # Create session in worker thread
                session = next(get_db())
                video_repo = SqlVideoRepository(session)
                schema_registry = SchemaRegistry()
                projection_sync = ProjectionSyncService(session)
                artifact_repo = SqlArtifactRepository(
                    session, schema_registry, projection_sync
                )
                transcription_handler = TranscriptionTaskHandler(
                    artifact_repository=artifact_repo,
                    schema_registry=schema_registry,
                )
                return TranscriptionWorker(
                    transcription_handler=transcription_handler,
                    video_repository=video_repo,
                    model_profile=self.profile_name,
                )

            return create_transcription_worker
        elif self.config.task_type == TaskType.SCENE_DETECTION:
            from ..database.connection import get_db
            from ..domain.schema_registry import SchemaRegistry
            from ..repositories.artifact_repository import SqlArtifactRepository
            from ..repositories.video_repository import SqlVideoRepository
            from ..services.projection_sync_service import ProjectionSyncService
            from .scene_detection_service import SceneDetectionService

            # Get settings from task_settings (use defaults if not specified)
            threshold = self.task_settings.get("scene_detection_threshold", 0.4)
            min_scene_len = self.task_settings.get("scene_detection_min_length", 0.6)

            # Create scene detection worker with dependencies
            def create_scene_detection_worker():
                # Create session in worker thread
                session = next(get_db())
                video_repo = SqlVideoRepository(session)
                schema_registry = SchemaRegistry()
                projection_sync = ProjectionSyncService(session)
                artifact_repo = SqlArtifactRepository(
                    session, schema_registry, projection_sync
                )
                scene_service = SceneDetectionService(
                    threshold=threshold,
                    min_scene_len=min_scene_len,
                )
                return SceneDetectionWorker(
                    scene_service=scene_service,
                    artifact_repository=artifact_repo,
                    video_repository=video_repo,
                    model_profile=self.profile_name,
                )

            return create_scene_detection_worker
        elif self.config.task_type == TaskType.OBJECT_DETECTION:
            from ..database.connection import get_db
            from ..domain.schema_registry import SchemaRegistry
            from ..repositories.artifact_repository import SqlArtifactRepository
            from ..repositories.video_repository import SqlVideoRepository
            from ..services.projection_sync_service import ProjectionSyncService
            from .object_detection_task_handler import ObjectDetectionTaskHandler

            # Get settings from task_settings
            model_name = self.task_settings.get("object_detection_model", "yolov8n.pt")
            sample_rate = self.task_settings.get("frame_sampling_interval", 30)

            # Create object detection worker with dependencies
            def create_object_detection_worker():
                # Create session in worker thread
                session = next(get_db())
                video_repo = SqlVideoRepository(session)
                schema_registry = SchemaRegistry()
                projection_sync = ProjectionSyncService(session)
                artifact_repo = SqlArtifactRepository(
                    session, schema_registry, projection_sync
                )
                detection_handler = ObjectDetectionTaskHandler(
                    artifact_repository=artifact_repo,
                    schema_registry=schema_registry,
                    model_name=model_name,
                    sample_rate=sample_rate,
                )
                return ObjectDetectionWorker(
                    detection_handler=detection_handler,
                    video_repository=video_repo,
                    model_profile=self.profile_name,
                )

            return create_object_detection_worker
        elif self.config.task_type == TaskType.FACE_DETECTION:
            from ..database.connection import get_db
            from ..domain.schema_registry import SchemaRegistry
            from ..repositories.artifact_repository import SqlArtifactRepository
            from ..repositories.video_repository import SqlVideoRepository
            from ..services.projection_sync_service import ProjectionSyncService
            from .face_detection_task_handler import FaceDetectionTaskHandler

            # Get settings from task_settings
            model_name = self.task_settings.get(
                "face_detection_model", "yolov8n-face.pt"
            )
            sample_rate = self.task_settings.get("frame_sampling_interval", 30)

            # Create face detection worker with dependencies
            def create_face_detection_worker():
                # Create session in worker thread
                session = next(get_db())
                video_repo = SqlVideoRepository(session)
                schema_registry = SchemaRegistry()
                projection_sync = ProjectionSyncService(session)
                artifact_repo = SqlArtifactRepository(
                    session, schema_registry, projection_sync
                )
                detection_handler = FaceDetectionTaskHandler(
                    artifact_repository=artifact_repo,
                    schema_registry=schema_registry,
                    model_name=model_name,
                    sample_rate=sample_rate,
                )
                return FaceDetectionWorker(
                    detection_handler=detection_handler,
                    video_repository=video_repo,
                    model_profile=self.profile_name,
                )

            return create_face_detection_worker
        elif self.config.task_type == TaskType.OCR:
            from ..database.connection import get_db
            from ..domain.schema_registry import SchemaRegistry
            from ..repositories.artifact_repository import SqlArtifactRepository
            from ..repositories.video_repository import SqlVideoRepository
            from ..services.projection_sync_service import ProjectionSyncService
            from .ocr_task_handler import OcrTaskHandler

            # Get settings from task_settings
            languages = self.task_settings.get("ocr_languages", ["en"])
            sample_rate = self.task_settings.get("frame_sampling_interval", 30)
            gpu = self.task_settings.get("ocr_gpu", False)

            # Create OCR worker with dependencies
            def create_ocr_worker():
                # Create session in worker thread
                session = next(get_db())
                video_repo = SqlVideoRepository(session)
                schema_registry = SchemaRegistry()
                projection_sync = ProjectionSyncService(session)
                artifact_repo = SqlArtifactRepository(
                    session, schema_registry, projection_sync
                )
                ocr_handler = OcrTaskHandler(
                    artifact_repository=artifact_repo,
                    schema_registry=schema_registry,
                    languages=languages,
                    sample_rate=sample_rate,
                    gpu=gpu,
                )
                return OcrWorker(
                    ocr_handler=ocr_handler,
                    video_repository=video_repo,
                    model_profile=self.profile_name,
                )

            return create_ocr_worker
        elif self.config.task_type == TaskType.PLACE_DETECTION:
            from ..database.connection import get_db
            from ..domain.schema_registry import SchemaRegistry
            from ..repositories.artifact_repository import SqlArtifactRepository
            from ..repositories.video_repository import SqlVideoRepository
            from ..services.projection_sync_service import ProjectionSyncService
            from .place_detection_task_handler import PlaceDetectionTaskHandler

            # Get settings from task_settings
            model_name = self.task_settings.get(
                "place_detection_model", "resnet18_places365"
            )
            sample_rate = self.task_settings.get("frame_sampling_interval", 30)
            top_k = self.task_settings.get("place_detection_top_k", 5)

            # Create place detection worker with dependencies
            def create_place_detection_worker():
                # Create session in worker thread
                session = next(get_db())
                video_repo = SqlVideoRepository(session)
                schema_registry = SchemaRegistry()
                projection_sync = ProjectionSyncService(session)
                artifact_repo = SqlArtifactRepository(
                    session, schema_registry, projection_sync
                )
                detection_handler = PlaceDetectionTaskHandler(
                    artifact_repository=artifact_repo,
                    schema_registry=schema_registry,
                    model_name=model_name,
                    sample_rate=sample_rate,
                    top_k=top_k,
                )
                return PlaceDetectionWorker(
                    detection_handler=detection_handler,
                    video_repository=video_repo,
                    model_profile=self.profile_name,
                )

            return create_place_detection_worker
        else:
            # Generic workers for other task types - need to pass task_type
            task_type = self.config.task_type

            def create_generic_worker():
                return TaskWorker(task_type)

            return create_generic_worker


class WorkerPoolManager:
    """Manages multiple worker pools."""

    def __init__(
        self,
        orchestrator: TaskOrchestrator,
        task_settings: dict | None = None,
        profile_name: str = "balanced",
    ):
        self.orchestrator = orchestrator
        self.task_settings = task_settings or {}
        self.profile_name = profile_name  # Store profile name for model_profile
        self.pools: dict[TaskType, WorkerPool] = {}
        self.is_running = False

    def add_worker_pool(self, config: WorkerConfig) -> None:
        """Add a worker pool for a task type."""
        if config.task_type in self.pools:
            raise ValueError(f"Worker pool for {config.task_type.value} already exists")

        pool = WorkerPool(
            config, self.orchestrator, self.task_settings, self.profile_name
        )
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
            WorkerConfig(TaskType.OBJECT_DETECTION, 2, ResourceType.GPU, 3),
            WorkerConfig(TaskType.FACE_DETECTION, 2, ResourceType.GPU, 3),
            WorkerConfig(TaskType.OCR, 2, ResourceType.GPU, 3),
            WorkerConfig(TaskType.PLACE_DETECTION, 2, ResourceType.GPU, 3),
        ]

        for config in default_configs:
            self.add_worker_pool(config)
