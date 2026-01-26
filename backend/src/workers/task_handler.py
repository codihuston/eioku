"""Task handler for Worker Service job consumption and result listening.

This module implements two handlers:
1. process_ml_task() - Quick arq job that sends to ML Service (fire-and-forget)
2. process_ml_results() - Result processor that listens on Redis and processes artifacts

The pattern:
- process_ml_task() enqueues job to ML Service and returns immediately
- ML Service processes and pushes results to Redis key
- process_ml_results() listens on Redis key (BLPOP) and processes into PostgreSQL
"""

import asyncio
import json
import logging
from datetime import datetime

import redis.asyncio as redis
from sqlalchemy.orm import Session

from ..config.redis_config import get_redis_settings
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

# Redis result key timeout (0 = infinite, arq will handle cancellation)
REDIS_BLPOP_TIMEOUT = 0


async def process_ml_task(
    task_id: str,
    task_type: str,
    video_id: str,
    video_path: str,
    config: dict | None = None,
) -> dict:
    """Process an ML task by sending to ML Service (fire-and-forget).

    This handler is called by arq when a job is consumed from the jobs queue.
    It implements the non-blocking queue pattern:
    1. Pre-flight check: verify task status is not COMPLETED/CANCELLED
    2. Update task status to RUNNING in PostgreSQL
    3. Send job to ML Service (fire-and-forget, non-blocking)
    4. Return immediately (result processor handles artifact creation)

    The result processor listens on Redis key for ML Service results and
    processes them into PostgreSQL artifacts.

    Args:
        task_id: Unique task identifier
        task_type: Type of task (e.g., 'object_detection')
        video_id: Video identifier
        video_path: Path to video file
        config: Optional task configuration

    Returns:
        Dictionary with task_id and status

    Raises:
        asyncio.CancelledError: If task is cancelled via arq
        ValueError: If task is already COMPLETED or CANCELLED
        RuntimeError: If database or Redis operations fail
    """
    session = None
    try:
        # Get database session
        session = next(get_db())
        task_repo = SQLAlchemyTaskRepository(session)

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

        # Step 2: Send job to ML Service (fire-and-forget)
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
            logger.info(
                f"Task {task_id} sent to ML Service (fire-and-forget) "
                f"with job_id {job_id}"
            )
        finally:
            await job_producer.close()

        # Return immediately - result processor will handle artifact creation
        return {
            "task_id": task_id,
            "status": "sent_to_ml_service",
        }

    except asyncio.CancelledError:
        # Handle task cancellation via arq
        logger.warning(f"Task {task_id} was cancelled via arq")
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


async def process_ml_results(
    task_id: str,
    task_type: str,
    video_id: str,
) -> dict:
    """Process ML Service results from Redis and insert into PostgreSQL.

    This handler listens on Redis key for ML Service results and processes
    them into PostgreSQL artifacts. It's designed to run as a separate
    background task or arq job.

    Args:
        task_id: Unique task identifier
        task_type: Type of task (e.g., 'object_detection')
        video_id: Video identifier

    Returns:
        Dictionary with task_id, status, and artifact_count

    Raises:
        asyncio.CancelledError: If task is cancelled via arq
        RuntimeError: If Redis or database operations fail
    """
    session = None
    redis_client = None
    try:
        # Get database session
        session = next(get_db())
        task_repo = SQLAlchemyTaskRepository(session)
        schema_registry = SchemaRegistry()
        projection_sync = ProjectionSyncService(session)
        artifact_repo = SqlArtifactRepository(session, schema_registry, projection_sync)

        logger.info(
            f"Waiting for ML results for task {task_id} ({task_type}) "
            f"on video {video_id}"
        )

        # Connect to Redis
        redis_settings = get_redis_settings()
        redis_client = await redis.from_url(redis_settings.url)

        result_key = f"ml_result:{task_id}"
        logger.info(f"Listening on Redis key {result_key} for ML Service results")

        # BLPOP with infinite timeout (0 = infinite, arq handles cancellation)
        result_data = await redis_client.blpop(result_key, timeout=REDIS_BLPOP_TIMEOUT)

        if not result_data:
            raise RuntimeError(f"No result received for task {task_id}")

        # result_data is (key, value) tuple
        _, result_json = result_data
        ml_response = json.loads(result_json)
        logger.info(f"Received ML Service result for task {task_id}")

        # Process results into PostgreSQL artifacts
        artifact_count = await _process_ml_results(
            task_id=task_id,
            task_type=task_type,
            video_id=video_id,
            ml_response=ml_response,
            session=session,
            artifact_repo=artifact_repo,
        )

        # Update task status to COMPLETED
        task = task_repo.find_by_video_and_type(video_id, task_type)
        if task:
            task_obj = task[0]
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
        # Handle task cancellation via arq
        logger.warning(f"Result processor for task {task_id} was cancelled via arq")
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
        logger.error(
            f"Error processing ML results for task {task_id}: {e}", exc_info=True
        )
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
        # Close Redis connection
        if redis_client:
            await redis_client.close()
        # Close database session
        if session:
            session.close()


async def _process_ml_results(
    task_id: str,
    task_type: str,
    video_id: str,
    ml_response: dict,
    session: Session,
    artifact_repo: SqlArtifactRepository,
) -> int:
    """Process ML Service results and insert artifacts into PostgreSQL.

    This function transforms the ML Service response into artifact records
    and inserts them into the database.

    Args:
        task_id: Task identifier
        task_type: Type of task
        video_id: Video identifier (asset_id)
        ml_response: ML Service response dictionary
        session: SQLAlchemy session for database access
        artifact_repo: Artifact repository for insertion

    Returns:
        Number of artifacts inserted

    Raises:
        ValueError: If ml_response is invalid
        RuntimeError: If database operations fail
    """
    if not ml_response:
        raise ValueError("ml_response cannot be empty")

    logger.info(f"Processing ML results for task {task_id} ({task_type})")

    # Extract detections/segments from response
    detections = ml_response.get("detections", [])
    if not detections and task_type != "scene_detection":
        logger.warning(
            f"No detections found in ML response for task {task_id} ({task_type})"
        )
        return 0

    # Handle scene detection separately (uses 'scenes' field)
    if task_type == "scene_detection":
        detections = ml_response.get("scenes", [])

    # Transform each detection to an artifact and insert
    artifact_count = 0
    for idx, detection in enumerate(detections):
        try:
            # Extract time span (in milliseconds)
            span_start_ms = int(detection.get("start_ms", 0))
            span_end_ms = int(detection.get("end_ms", 0))

            # Validate time span
            if span_start_ms < 0 or span_end_ms < 0:
                logger.warning(
                    f"Invalid time span for detection {idx}: "
                    f"start={span_start_ms}, end={span_end_ms}"
                )
                continue

            if span_start_ms > span_end_ms:
                logger.warning(
                    f"Invalid time span for detection {idx}: "
                    f"start > end ({span_start_ms} > {span_end_ms})"
                )
                continue

            # Create artifact dictionary
            artifact = {
                "task_id": task_id,
                "video_id": video_id,
                "task_type": task_type,
                "span_start_ms": span_start_ms,
                "span_end_ms": span_end_ms,
                "payload": detection,
                "config_hash": ml_response.get("config_hash", ""),
                "input_hash": ml_response.get("input_hash", ""),
                "run_id": ml_response.get("run_id", ""),
                "producer": ml_response.get("producer", "ml-service"),
                "producer_version": ml_response.get("producer_version", "1.0.0"),
                "model_profile": ml_response.get("model_profile", "balanced"),
            }

            # Insert artifact into database
            artifact_repo.create(artifact)
            artifact_count += 1
            logger.debug(f"Inserted artifact {idx} for task {task_id}")

        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error processing detection {idx} for task {task_id}: {e}")
            continue

    logger.info(
        f"Processed {artifact_count} detections to artifacts "
        f"for task {task_id} ({task_type})"
    )
    return artifact_count
