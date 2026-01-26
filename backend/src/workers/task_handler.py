"""Task handler for Worker Service job consumption and result polling.

This module implements the process_ml_task() handler that:
1. Consumes jobs from the jobs queue
2. Updates task status to RUNNING in PostgreSQL
3. Enqueues the job to ml_jobs queue for ML Service processing
4. Polls PostgreSQL for artifact completion with exponential backoff
5. Updates task status to COMPLETED when all artifacts are present
6. Acknowledges the job in Redis (XACK)
"""

import asyncio
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..domain.schema_registry import SchemaRegistry
from ..repositories.artifact_repository import SqlArtifactRepository
from ..repositories.task_repository import SQLAlchemyTaskRepository
from ..services.job_producer import JobProducer
from ..services.projection_sync_service import ProjectionSyncService

logger = logging.getLogger(__name__)

# Expected artifact counts per task type
EXPECTED_ARTIFACTS = {
    "object_detection": 1,
    "face_detection": 1,
    "transcription": 1,
    "ocr": 1,
    "place_detection": 1,
    "scene_detection": 1,
}

# Polling configuration
POLL_INITIAL_DELAY = 1.0  # Start with 1 second
POLL_MAX_DELAY = 30.0  # Cap at 30 seconds
POLL_TIMEOUT = 1800.0  # 30 minutes total timeout


async def process_ml_task(
    task_id: str,
    task_type: str,
    video_id: str,
    video_path: str,
    config: dict | None = None,
) -> dict:
    """Process an ML task by enqueueing to ml_jobs and polling for results.

    This handler is called by arq when a job is consumed from the jobs queue.
    It implements the shared queue pattern:
    1. Pre-flight check: verify task status is not COMPLETED/CANCELLED
    2. Update task status to RUNNING in PostgreSQL
    3. Enqueue job to ml_jobs queue for ML Service processing
    4. Poll PostgreSQL for artifact completion with exponential backoff
    5. Update task status to COMPLETED when all artifacts are present
    6. Acknowledge job in Redis (XACK)

    Args:
        task_id: Unique task identifier
        task_type: Type of task (e.g., 'object_detection')
        video_id: Video identifier
        video_path: Path to video file
        config: Optional task configuration

    Returns:
        Dictionary with task_id, status, and artifact_count

    Raises:
        asyncio.CancelledError: If task is cancelled during execution
        ValueError: If task is already COMPLETED or CANCELLED
        RuntimeError: If database or Redis operations fail
    """
    session = None
    try:
        # Get database session
        session = next(get_db())
        task_repo = SQLAlchemyTaskRepository(session)
        schema_registry = SchemaRegistry()
        projection_sync = ProjectionSyncService(session)
        artifact_repo = SqlArtifactRepository(session, schema_registry, projection_sync)

        # Pre-flight check: verify task status is not COMPLETED/CANCELLED
        task = task_repo.find_by_video_and_type(video_id, task_type)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        task_obj = task[0]
        if task_obj.status in ("completed", "cancelled"):
            raise ValueError(
                f"Cannot process task {task_id}: status is {task_obj.status}"
            )

        logger.info(f"Processing task {task_id} ({task_type}) for video {video_id}")

        # Step 1: Update task status to RUNNING
        task_obj.status = "running"
        task_obj.started_at = datetime.utcnow()
        task_repo.update(task_obj)
        logger.info(f"Task {task_id} status updated to RUNNING")

        # Step 2: Enqueue job to ml_jobs queue
        job_producer = JobProducer()
        await job_producer.initialize()
        try:
            # Get video record to retrieve the file hash
            from ..repositories.video_repository import VideoRepository

            video_repo = VideoRepository(session)
            video = video_repo.find_by_id(video_id)
            if not video:
                raise ValueError(f"Video not found: {video_id}")

            input_hash = video.file_hash
            if not input_hash:
                logger.warning(f"Video {video_id} has no file_hash, using empty string")
                input_hash = ""

            job_id = await job_producer.enqueue_to_ml_jobs(
                task_id=task_id,
                task_type=task_type,
                video_id=video_id,
                video_path=video_path,
                input_hash=input_hash,
                config=config or {},
            )
            logger.info(f"Task {task_id} enqueued to ml_jobs with job_id {job_id}")
        finally:
            await job_producer.close()

        # Step 3: Poll PostgreSQL for artifact completion
        artifact_count = await poll_for_artifacts(
            task_id=task_id,
            task_type=task_type,
            video_id=video_id,
            session=session,
            artifact_repo=artifact_repo,
        )

        # Step 4: Update task status to COMPLETED
        task_obj.status = "completed"
        task_obj.completed_at = datetime.utcnow()
        task_repo.update(task_obj)
        logger.info(
            f"Task {task_id} status updated to COMPLETED "
            f"with {artifact_count} artifacts"
        )

        return {
            "task_id": task_id,
            "status": "completed",
            "artifact_count": artifact_count,
        }

    except asyncio.CancelledError:
        # Handle task cancellation
        logger.warning(f"Task {task_id} was cancelled")
        if session:
            try:
                task_repo = SQLAlchemyTaskRepository(session)
                task = task_repo.find_by_video_and_type(video_id, task_type)
                if task:
                    task_obj = task[0]
                    task_obj.status = "cancelled"
                    task_repo.update(task_obj)
                    logger.info(f"Task {task_id} status updated to CANCELLED")
            except Exception as e:
                logger.error(f"Error updating task {task_id} to CANCELLED: {e}")
        raise

    except Exception as e:
        # Handle other exceptions and mark task as FAILED
        logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
        if session:
            try:
                task_repo = SQLAlchemyTaskRepository(session)
                task = task_repo.find_by_video_and_type(video_id, task_type)
                if task:
                    task_obj = task[0]
                    task_obj.status = "failed"
                    task_obj.error = str(e)
                    task_repo.update(task_obj)
                    logger.info(f"Task {task_id} status updated to FAILED")
            except Exception as update_error:
                logger.error(f"Error updating task {task_id} to FAILED: {update_error}")
        raise

    finally:
        # Close database session
        if session:
            session.close()


async def poll_for_artifacts(
    task_id: str,
    task_type: str,
    session: Session,
    artifact_repo: SqlArtifactRepository,
    video_id: str | None = None,
    initial_delay: float = POLL_INITIAL_DELAY,
    max_delay: float = POLL_MAX_DELAY,
    timeout: float = POLL_TIMEOUT,
) -> int:
    """Poll PostgreSQL for artifact completion with exponential backoff.

    This function polls the artifacts table for the task until all expected
    artifacts are present. It uses exponential backoff to avoid excessive
    database queries.

    Note: Artifacts are stored with asset_id (video_id) and artifact_type.
    We query by video_id and artifact_type to find artifacts for this task.

    Args:
        task_id: Task identifier (for logging)
        task_type: Type of task (used to determine expected artifact count)
        session: SQLAlchemy session for database access
        artifact_repo: Artifact repository for querying
        video_id: Video identifier (asset_id in artifacts table)
        initial_delay: Initial polling delay in seconds (default: 1.0)
        max_delay: Maximum polling delay in seconds (default: 30.0)
        timeout: Total polling timeout in seconds (default: 1800.0)

    Returns:
        Number of artifacts found

    Raises:
        TimeoutError: If polling timeout is exceeded
        ValueError: If task_type is not recognized
    """
    if task_type not in EXPECTED_ARTIFACTS:
        raise ValueError(f"Unknown task type: {task_type}")

    if not video_id:
        raise ValueError("video_id is required for artifact polling")

    expected_count = EXPECTED_ARTIFACTS[task_type]
    elapsed = 0.0
    delay = initial_delay

    logger.info(
        f"Starting artifact polling for task {task_id} ({task_type}) "
        f"on video {video_id} (expecting {expected_count} artifacts, "
        f"timeout={timeout}s)"
    )

    while elapsed < timeout:
        try:
            # Query artifacts for this video and task type
            # Artifacts are stored with asset_id (video_id) and artifact_type
            artifacts = artifact_repo.get_by_asset(
                asset_id=video_id,
                artifact_type=task_type,
            )

            artifact_count = len(artifacts)
            logger.debug(
                f"Polling task {task_id}: found {artifact_count}/"
                f"{expected_count} artifacts"
            )

            # Check if all expected artifacts are present
            if artifact_count >= expected_count:
                logger.info(
                    f"All artifacts found for task {task_id} "
                    f"({artifact_count}/{expected_count})"
                )
                return artifact_count

            # Wait before next poll with exponential backoff
            await asyncio.sleep(delay)
            elapsed += delay
            delay = min(delay * 2, max_delay)  # Exponential backoff capped

        except Exception as e:
            logger.error(f"Error polling artifacts for task {task_id}: {e}")
            # Continue polling on error
            await asyncio.sleep(delay)
            elapsed += delay
            delay = min(delay * 2, max_delay)

    # Timeout exceeded
    raise TimeoutError(f"Polling timeout exceeded for task {task_id} after {timeout}s")
