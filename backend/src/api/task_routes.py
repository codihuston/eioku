"""Task processing and status API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..repositories.task_repository import SQLAlchemyTaskRepository
from ..repositories.video_repository import SqlVideoRepository
from ..services.job_producer import JobProducer
from ..services.task_orchestrator import TaskOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


# ============================================================================
# Request/Response Models for OpenAPI Documentation
# ============================================================================


class EnqueueTaskResponse(BaseModel):
    """Response model for task enqueueing endpoint."""

    task_id: str = Field(..., description="The unique identifier of the task")
    job_id: str = Field(..., description="The job ID in Redis (format: ml_{task_id})")
    status: str = Field(
        ..., description="Status of the enqueueing operation", example="enqueued"
    )
    task_type: str = Field(
        ..., description="Type of ML task", example="object_detection"
    )
    video_id: str = Field(..., description="The video ID associated with this task")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "job_id": "ml_550e8400-e29b-41d4-a716-446655440000",
                "status": "enqueued",
                "task_type": "object_detection",
                "video_id": "550e8400-e29b-41d4-a716-446655440001",
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {"detail": "Task 550e8400-e29b-41d4-a716-446655440000 not found"}
        }


class CancelTaskResponse(BaseModel):
    """Response model for task cancellation endpoint."""

    task_id: str = Field(..., description="The unique identifier of the task")
    status: str = Field(
        ..., description="Status of the cancellation operation", example="cancelled"
    )
    message: str = Field(..., description="Cancellation message")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "cancelled",
                "message": (
                    "Task cancelled successfully. "
                    "Note: If ML inference is already running, it will complete."
                ),
            }
        }


class RetryTaskResponse(BaseModel):
    """Response model for task retry endpoint."""

    task_id: str = Field(..., description="The unique identifier of the task")
    job_id: str = Field(..., description="The new job ID in Redis")
    status: str = Field(
        ..., description="Status of the retry operation", example="pending"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "job_id": "ml_550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
            }
        }


class TaskListResponse(BaseModel):
    """Response model for task list endpoint."""

    tasks: list[dict] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks matching filters")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "tasks": [
                    {
                        "task_id": "550e8400-e29b-41d4-a716-446655440000",
                        "task_type": "object_detection",
                        "status": "completed",
                        "video_id": "550e8400-e29b-41d4-a716-446655440001",
                        "created_at": "2024-01-25T10:00:00",
                        "started_at": "2024-01-25T10:00:05",
                        "completed_at": "2024-01-25T10:05:00",
                    }
                ],
                "total": 42,
                "limit": 10,
                "offset": 0,
            }
        }


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "/{task_id}/enqueue",
    response_model=EnqueueTaskResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task or video not found"},
        400: {
            "model": ErrorResponse,
            "description": "Task not in PENDING status",
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Manually enqueue a task for processing",
    description="Enqueue a task that is in PENDING status to the job queue. "
    "The task must exist and be in PENDING status to be enqueued successfully.",
)
async def enqueue_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> EnqueueTaskResponse:
    """Manually enqueue a task for processing.

    This endpoint allows manual enqueueing of a task that is in PENDING status.
    The task must exist and be in PENDING status to be enqueued.

    **Requirements**: 1.3

    Args:
        task_id: The task ID to enqueue
        db: Database session

    Returns:
        EnqueueTaskResponse with job_id, task_id, and status

    Raises:
        HTTPException 404: If task or video not found
        HTTPException 400: If task not in PENDING status
        HTTPException 500: If enqueueing fails
    """
    try:
        # Get task from database
        task_repo = SQLAlchemyTaskRepository(db)
        task = task_repo.find_by_id(task_id)

        if not task:
            logger.warning(f"Task not found: {task_id}")
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Verify task is in PENDING status
        if task.status != "pending":
            logger.warning(f"Cannot enqueue task {task_id} in status {task.status}")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot enqueue task in status {task.status}. "
                f"Task must be in PENDING status.",
            )

        # Get video details
        video_repo = SqlVideoRepository(db)
        video = video_repo.find_by_id(task.video_id)

        if not video:
            logger.error(f"Video not found for task {task_id}: {task.video_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Video {task.video_id} not found",
            )

        # Get default config for task type
        from ..services.video_discovery_service import VideoDiscoveryService

        discovery_service = VideoDiscoveryService(None, video_repo)
        config = discovery_service._get_default_config(task.task_type)

        # Initialize JobProducer and enqueue task
        job_producer = JobProducer()
        await job_producer.initialize()

        try:
            job_id = await job_producer.enqueue_task(
                task_id=task_id,
                task_type=task.task_type,
                video_id=str(task.video_id),
                video_path=video.file_path,
                config=config,
            )

            logger.info(
                f"Successfully enqueued task {task_id} ({task.task_type}) "
                f"with job_id {job_id}"
            )

            return EnqueueTaskResponse(
                task_id=task_id,
                job_id=job_id,
                status="enqueued",
                task_type=task.task_type,
                video_id=str(task.video_id),
            )

        finally:
            await job_producer.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enqueue task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue task: {str(e)}",
        )


@router.post(
    "/{task_id}/cancel",
    response_model=CancelTaskResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        400: {
            "model": ErrorResponse,
            "description": "Task cannot be cancelled in current status",
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Cancel a task",
    description="Cancel a task that is in PENDING or RUNNING status. "
    "**LIMITATION**: If ML inference is already running in the ML Service, "
    "it will continue to completion and persist results. Cancellation is best-effort.",
)
async def cancel_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> CancelTaskResponse:
    """Cancel a task that is in PENDING or RUNNING status.

    This endpoint marks a task as CANCELLED and attempts to abort the job in Redis.

    **IMPORTANT LIMITATION**: If the ML Service has already started processing the
    task, the inference will continue to completion and results will be persisted to
    PostgreSQL. This is because:
    1. ML inference operations cannot be safely interrupted mid-execution
    2. The ML Service worker doesn't have a cancellation mechanism
    3. Aborting the job in Redis only prevents re-queuing, not execution

    To implement hard cancellation, the ML Service would need to:
    - Check a cancellation flag in Redis periodically during inference
    - Gracefully stop processing if cancelled
    - Skip artifact persistence if cancelled

    **Requirements**: 10.1, 10.2, 10.3, 10.4

    Args:
        task_id: The task ID to cancel
        db: Database session

    Returns:
        CancelTaskResponse with task_id, status, and message

    Raises:
        HTTPException 404: If task not found
        HTTPException 400: If task cannot be cancelled in current status
        HTTPException 500: If cancellation fails
    """
    try:
        # Get task from database
        task_repo = SQLAlchemyTaskRepository(db)
        task = task_repo.find_by_id(task_id)

        if not task:
            logger.warning(f"Task not found: {task_id}")
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Verify task can be cancelled (PENDING or RUNNING)
        if task.status not in ("pending", "running"):
            logger.warning(f"Cannot cancel task {task_id} in status {task.status}")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task in status {task.status}. "
                f"Task must be in PENDING or RUNNING status.",
            )

        # Mark task as CANCELLED in PostgreSQL
        task.status = "cancelled"
        task_repo.update(task)

        logger.info(f"Task {task_id} marked as CANCELLED in PostgreSQL")

        # Attempt to abort job in Redis (best-effort)
        try:
            from arq import create_pool

            from ..config.redis_config import get_redis_settings

            redis_settings = get_redis_settings()
            pool = await create_pool(redis_settings)

            try:
                job_id = f"ml_{task_id}"
                job = await pool.job(job_id)

                if job:
                    await job.abort()
                    logger.info(f"Job {job_id} aborted in Redis")
                else:
                    logger.debug(
                        f"Job {job_id} not found in Redis (may have completed)"
                    )

            finally:
                await pool.close()

        except Exception as e:
            logger.warning(
                f"Failed to abort job in Redis for task {task_id}: {e}. "
                f"Task is still marked as CANCELLED in PostgreSQL."
            )

        return CancelTaskResponse(
            task_id=task_id,
            status="cancelled",
            message="Task cancelled successfully. "
            "Note: If ML inference is already running, it will complete.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}",
        )


@router.post(
    "/{task_id}/retry",
    response_model=RetryTaskResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        400: {
            "model": ErrorResponse,
            "description": "Task cannot be retried in current status",
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Retry a failed or cancelled task",
    description="Reset a task to PENDING status and re-enqueue it for processing. "
    "The task must be in FAILED or CANCELLED status.",
)
async def retry_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> RetryTaskResponse:
    """Retry a failed or cancelled task.

    This endpoint resets a task to PENDING status and re-enqueues it for processing.

    **Requirements**: 10.1

    Args:
        task_id: The task ID to retry
        db: Database session

    Returns:
        RetryTaskResponse with task_id, job_id, and status

    Raises:
        HTTPException 404: If task not found
        HTTPException 400: If task cannot be retried in current status
        HTTPException 500: If retry fails
    """
    try:
        # Get task from database
        task_repo = SQLAlchemyTaskRepository(db)
        task = task_repo.find_by_id(task_id)

        if not task:
            logger.warning(f"Task not found: {task_id}")
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Verify task is in FAILED or CANCELLED status
        if task.status not in ("failed", "cancelled"):
            logger.warning(f"Cannot retry task {task_id} in status {task.status}")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry task in status {task.status}. "
                f"Task must be in FAILED or CANCELLED status.",
            )

        # Get video details
        video_repo = SqlVideoRepository(db)
        video = video_repo.find_by_id(task.video_id)

        if not video:
            logger.error(f"Video not found for task {task_id}: {task.video_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Video {task.video_id} not found",
            )

        # Get default config for task type
        from ..services.video_discovery_service import VideoDiscoveryService

        discovery_service = VideoDiscoveryService(None, video_repo)
        config = discovery_service._get_default_config(task.task_type)

        # Reset task to PENDING
        task.status = "pending"
        task.started_at = None
        task.completed_at = None
        task.error = None
        task_repo.update(task)

        logger.info(f"Task {task_id} reset to PENDING status")

        # Re-enqueue task
        job_producer = JobProducer()
        await job_producer.initialize()

        try:
            job_id = await job_producer.enqueue_task(
                task_id=task_id,
                task_type=task.task_type,
                video_id=str(task.video_id),
                video_path=video.file_path,
                config=config,
            )

            logger.info(
                f"Successfully retried task {task_id} ({task.task_type}) "
                f"with job_id {job_id}"
            )

            return RetryTaskResponse(
                task_id=task_id,
                job_id=job_id,
                status="pending",
            )

        finally:
            await job_producer.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retry task: {str(e)}",
        )


@router.get(
    "",
    response_model=TaskListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="List tasks with filtering and sorting",
    description="Get a paginated list of tasks with optional filtering by status, "
    "task_type, or video_id, and sorting by created_at, started_at, or running_time.",
)
async def list_tasks(
    status: str | None = None,
    task_type: str | None = None,
    video_id: str | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> TaskListResponse:
    """List tasks with filtering and sorting.

    This endpoint returns a paginated list of tasks with optional filtering
    and sorting capabilities.

    **Requirements**: 10.7, 10.8

    Args:
        status: Filter by task status (pending, running, completed, failed, cancelled)
        task_type: Filter by task type (object_detection, face_detection, etc.)
        video_id: Filter by video ID
        sort_by: Sort field (created_at, started_at, running_time)
        sort_order: Sort order (asc, desc)
        limit: Number of tasks to return (default: 10, max: 100)
        offset: Pagination offset (default: 0)
        db: Database session

    Returns:
        TaskListResponse with tasks list and pagination info

    Raises:
        HTTPException 400: If invalid parameters provided
        HTTPException 500: If query fails
    """
    try:
        # Validate parameters
        valid_statuses = {"pending", "running", "completed", "failed", "cancelled"}
        if status and status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}",
            )

        valid_sort_fields = {"created_at", "started_at", "running_time"}
        if sort_by not in valid_sort_fields:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid sort_by. Must be one of: "
                    f"{', '.join(valid_sort_fields)}"
                ),
            )

        if sort_order not in ("asc", "desc"):
            raise HTTPException(
                status_code=400,
                detail="Invalid sort_order. Must be 'asc' or 'desc'",
            )

        # Limit pagination
        limit = min(limit, 100)
        if limit < 1:
            limit = 10
        if offset < 0:
            offset = 0

        # Get task repository
        task_repo = SQLAlchemyTaskRepository(db)

        # Build query filters
        filters = {}
        if status:
            filters["status"] = status
        if task_type:
            filters["task_type"] = task_type
        if video_id:
            filters["video_id"] = video_id

        # Get all tasks matching filters (we'll sort in Python for simplicity)
        all_tasks = []

        if status:
            all_tasks = task_repo.find_by_status(status)
        else:
            # Get all tasks - this is a limitation of the current repository
            # In production, you'd want a more efficient query
            all_tasks = task_repo.find_all() if hasattr(task_repo, "find_all") else []

        # Apply additional filters
        if task_type:
            all_tasks = [t for t in all_tasks if t.task_type == task_type]
        if video_id:
            all_tasks = [t for t in all_tasks if str(t.video_id) == video_id]

        # Sort tasks
        def get_sort_key(task):
            if sort_by == "created_at":
                return task.created_at or ""
            elif sort_by == "started_at":
                return task.started_at or ""
            elif sort_by == "running_time":
                if hasattr(task, "started_at") and hasattr(task, "completed_at"):
                    if task.started_at and task.completed_at:
                        return (task.completed_at - task.started_at).total_seconds()
                return 0
            return ""

        all_tasks.sort(key=get_sort_key, reverse=(sort_order == "desc"))

        # Apply pagination
        total = len(all_tasks)
        paginated_tasks = all_tasks[offset : offset + limit]

        # Format response
        tasks_data = [
            {
                "task_id": str(task.task_id),
                "task_type": task.task_type,
                "status": task.status,
                "video_id": str(task.video_id),
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat()
                if hasattr(task, "started_at") and task.started_at
                else None,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
                "error": getattr(task, "error", None),
            }
            for task in paginated_tasks
        ]

        logger.info(
            f"Listed {len(paginated_tasks)} tasks (total: {total}, "
            f"offset: {offset}, limit: {limit})"
        )

        return TaskListResponse(
            tasks=tasks_data,
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tasks: {str(e)}",
        )


@router.post("/process")
async def trigger_task_processing(db: Session = Depends(get_db)) -> dict:
    """Manually trigger task processing for discovered videos."""
    try:
        # Initialize repositories and orchestrator
        video_repo = SqlVideoRepository(db)
        task_repo = SQLAlchemyTaskRepository(db)
        orchestrator = TaskOrchestrator(task_repo, video_repo)

        # Process discovered videos
        discovered_count = orchestrator.process_discovered_videos()
        hashed_count = orchestrator.process_hashed_videos()

        logger.info(
            f"Task processing triggered: {discovered_count} discovered, "
            f"{hashed_count} hashed"
        )

        return {
            "status": "success",
            "discovered_videos_processed": discovered_count,
            "hashed_videos_processed": hashed_count,
            "message": f"Created tasks for {discovered_count + hashed_count} videos",
        }

    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task processing failed: {e}")


@router.post("/create-transcription-task")
async def create_transcription_task(
    video_id: str, db: Session = Depends(get_db)
) -> dict:
    """Create a transcription task for a hashed video."""
    try:
        import uuid
        from datetime import datetime

        from sqlalchemy import text

        # Check if video exists and is hashed
        video_result = db.execute(
            text(
                "SELECT * FROM videos WHERE video_id = :video_id AND status = 'hashed'"
            ),
            {"video_id": video_id},
        ).fetchone()

        if not video_result:
            return {"status": "error", "message": "Video not found or not hashed"}

        # Create transcription task
        task_id = str(uuid.uuid4())
        db.execute(
            text(
                """
            INSERT INTO tasks (task_id, video_id, task_type, status, priority,
                              dependencies, created_at)
            VALUES (:task_id, :video_id, 'transcription', 'pending', 1, '[]',
                    :created_at)
        """
            ),
            {
                "task_id": task_id,
                "video_id": video_id,
                "created_at": datetime.utcnow(),
            },
        )
        db.commit()

        logger.info(f"Created transcription task {task_id} for video {video_id}")
        return {"message": f"Created transcription task: {task_id}"}

    except Exception as e:
        logger.error(f"Failed to create transcription task: {e}")
        return {"status": "error", "message": str(e)}


async def get_task_status(db: Session = Depends(get_db)) -> dict:
    """Get current task processing status."""
    try:
        video_repo = SqlVideoRepository(db)
        task_repo = SQLAlchemyTaskRepository(db)

        # Get video counts by status
        discovered_videos = video_repo.find_by_status("discovered")
        hashed_videos = video_repo.find_by_status("hashed")
        processing_videos = video_repo.find_by_status("processing")
        completed_videos = video_repo.find_by_status("completed")
        failed_videos = video_repo.find_by_status("failed")

        video_status_counts = {
            "discovered": len(discovered_videos),
            "hashed": len(hashed_videos),
            "processing": len(processing_videos),
            "completed": len(completed_videos),
            "failed": len(failed_videos),
        }

        # Get task counts by status
        pending_tasks = task_repo.find_by_status("pending")
        running_tasks = task_repo.find_by_status("running")
        completed_tasks = task_repo.find_by_status("completed")
        failed_tasks = task_repo.find_by_status("failed")

        task_counts = {
            "pending": len(pending_tasks),
            "running": len(running_tasks),
            "completed": len(completed_tasks),
            "failed": len(failed_tasks),
        }

        # Get recent tasks (last 10 pending tasks)
        recent_task_info = [
            {
                "task_id": task.task_id,
                "video_id": task.video_id,
                "task_type": task.task_type,
                "status": task.status,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
            }
            for task in pending_tasks[:10]  # Show first 10 pending tasks
        ]

        total_videos = sum(video_status_counts.values())
        total_tasks = sum(task_counts.values())

        return {
            "video_status_counts": video_status_counts,
            "task_counts": task_counts,
            "recent_tasks": recent_task_info,
            "total_videos": total_videos,
            "total_tasks": total_tasks,
        }

    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {e}")


@router.get("/videos/{video_id}/tasks")
async def get_video_tasks(video_id: str, db: Session = Depends(get_db)) -> list[dict]:
    """Get all tasks for a specific video."""
    try:
        task_repo = SQLAlchemyTaskRepository(db)
        tasks = task_repo.find_by_video_id(video_id)

        return [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat()
                if hasattr(task, "started_at") and task.started_at
                else None,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
                "error_message": getattr(task, "error_message", None),
            }
            for task in tasks
        ]

    except Exception as e:
        logger.error(f"Failed to get tasks for video {video_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get tasks for video: {e}"
        )


@router.post("/create-hash-task")
async def create_hash_task(db: Session = Depends(get_db)) -> dict:
    """Create a hash task for discovered videos."""
    try:
        import uuid
        from datetime import datetime

        from sqlalchemy import text

        # Get discovered videos
        video_result = db.execute(
            text(
                """
            SELECT video_id, filename, file_path
            FROM videos
            WHERE status = 'discovered'
        """
            )
        ).fetchone()

        if not video_result:
            return {"status": "error", "message": "No discovered videos found"}

        video_id, filename, file_path = video_result

        # Create hash task
        task_id = str(uuid.uuid4())

        db.execute(
            text(
                """
            INSERT INTO tasks (task_id, video_id, task_type, status, priority,
                              dependencies, created_at)
            VALUES (:task_id, :video_id, 'hash', 'pending', 1, '[]', :created_at)
        """
            ),
            {
                "task_id": task_id,
                "video_id": video_id,
                "created_at": datetime.utcnow(),
            },
        )

        db.commit()

        return {
            "status": "success",
            "message": f"Created hash task for {filename}",
            "task_id": task_id,
            "video_id": video_id,
            "filename": filename,
        }

    except Exception as e:
        logger.error(f"Failed to create hash task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create hash task: {e}")


@router.post("/process-pending")
async def process_pending_tasks(db: Session = Depends(get_db)) -> dict:
    """Manually process pending tasks (for development/testing)."""
    try:
        video_repo = SqlVideoRepository(db)
        task_repo = SQLAlchemyTaskRepository(db)

        # Get pending tasks
        pending_tasks = task_repo.find_by_status("pending")

        if not pending_tasks:
            return {
                "status": "success",
                "message": "No pending tasks to process",
                "processed_tasks": 0,
            }

        processed_count = 0
        results = []

        for task in pending_tasks:
            try:
                if task.task_type == "hash":
                    # Process hash task
                    from ..services.file_hash_service import FileHashService
                    from ..services.worker_pool_manager import HashWorker

                    hash_service = FileHashService()
                    HashWorker(hash_service=hash_service)

                    # Get video
                    video = video_repo.find_by_id(task.video_id)
                    if not video:
                        raise Exception(f"Video {task.video_id} not found")

                    logger.info(f"Processing hash task for {video.filename}")

                    # Calculate hash using actual video file path
                    hash_result = hash_service.calculate_hash(video.file_path)

                    # Update task status (using direct SQL to avoid repository issues)
                    from sqlalchemy import text

                    db.execute(
                        text(
                            """
                        UPDATE tasks
                        SET status = 'completed',
                            completed_at = datetime('now')
                        WHERE task_id = :task_id
                    """
                        ),
                        {"task_id": task.task_id},
                    )

                    # Update video status and hash
                    db.execute(
                        text(
                            """
                        UPDATE videos
                        SET status = 'hashed',
                            file_hash = :file_hash,
                            updated_at = datetime('now')
                        WHERE video_id = :video_id
                    """
                        ),
                        {"video_id": video.video_id, "file_hash": hash_result},
                    )

                    db.commit()

                    results.append(
                        {
                            "task_id": task.task_id,
                            "task_type": task.task_type,
                            "video_id": task.video_id,
                            "status": "completed",
                            "result": hash_result,
                        }
                    )

                    processed_count += 1
                    logger.info(
                        f"Completed hash task for {video.filename}: {hash_result}"
                    )

                else:
                    logger.warning(f"Unsupported task type: {task.task_type}")

            except Exception as e:
                logger.error(f"Failed to process task {task.task_id}: {e}")

                # Mark task as failed
                from sqlalchemy import text

                db.execute(
                    text(
                        """
                    UPDATE tasks
                    SET status = 'failed',
                        error = :error_msg
                    WHERE task_id = :task_id
                """
                    ),
                    {"task_id": task.task_id, "error_msg": str(e)},
                )
                db.commit()

                results.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "video_id": task.video_id,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return {
            "status": "success",
            "message": f"Processed {processed_count} tasks",
            "processed_tasks": processed_count,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task processing failed: {e}")


@router.post("/cleanup")
async def cleanup_orphaned_tasks(db: Session = Depends(get_db)) -> dict:
    """Clean up tasks for videos that no longer exist."""
    try:
        video_repo = SqlVideoRepository(db)
        task_repo = SQLAlchemyTaskRepository(db)

        # Get all pending tasks
        pending_tasks = task_repo.find_by_status("pending")

        orphaned_count = 0
        for task in pending_tasks:
            # Check if video still exists
            video = video_repo.find_by_id(task.video_id)
            if not video or video.status == "missing":
                # Mark task as failed
                task.status = "failed"
                task.error_message = "Video no longer exists or is missing"
                task_repo.save(task)
                orphaned_count += 1
                logger.info(f"Marked orphaned task {task.task_id} as failed")

        return {
            "status": "success",
            "orphaned_tasks_cleaned": orphaned_count,
            "message": f"Cleaned up {orphaned_count} orphaned tasks",
        }

    except Exception as e:
        logger.error(f"Task cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task cleanup failed: {e}")


@router.get("/queue/status")
async def get_queue_status() -> dict:
    """Get current task queue status."""
    # This would need to be connected to the actual worker pool manager
    # For now, return a placeholder
    return {
        "message": "Queue status endpoint - needs worker pool integration",
        "queues": {
            "hash": {"pending": 0, "processing": 0},
            "transcription": {"pending": 0, "processing": 0},
            "scene_detection": {"pending": 0, "processing": 0},
            "object_detection": {"pending": 0, "processing": 0},
            "face_detection": {"pending": 0, "processing": 0},
        },
    }


@router.post("/startup")
async def startup_worker_pools() -> dict:
    """Start worker pools and store them globally."""
    try:
        from src.database.connection import get_db
        from src.repositories.task_repository import SQLAlchemyTaskRepository
        from src.repositories.video_repository import SqlVideoRepository
        from src.services.task_orchestration import TaskType
        from src.services.task_orchestrator import TaskOrchestrator
        from src.services.worker_pool_manager import (
            ResourceType,
            WorkerConfig,
            WorkerPoolManager,
        )

        session = next(get_db())
        try:
            video_repo = SqlVideoRepository(session)
            task_repo = SQLAlchemyTaskRepository(session)
            orchestrator = TaskOrchestrator(task_repo, video_repo)

            # Create global worker pool manager
            global_pool_manager = WorkerPoolManager(orchestrator)

            # Add worker pools
            hash_config = WorkerConfig(TaskType.HASH, 2, ResourceType.CPU, 1)
            global_pool_manager.add_worker_pool(hash_config)

            transcription_config = WorkerConfig(
                TaskType.TRANSCRIPTION, 1, ResourceType.CPU, 1
            )
            global_pool_manager.add_worker_pool(transcription_config)

            # Start all worker pools
            global_pool_manager.start_all()

            # Store globally (simple approach)
            import src.main_api as main_module

            main_module.global_pool_manager = global_pool_manager

            return {"status": "success", "message": "Worker pools started successfully"}

        finally:
            session.close()

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


async def test_worker_pools() -> dict:
    """Test endpoint to manually start worker pools."""
    try:
        print("🧪 Testing worker pools...")

        from src.database.connection import get_db
        from src.repositories.task_repository import SQLAlchemyTaskRepository
        from src.repositories.video_repository import SqlVideoRepository
        from src.services.task_orchestration import TaskType
        from src.services.task_orchestrator import TaskOrchestrator
        from src.services.worker_pool_manager import (
            ResourceType,
            WorkerConfig,
            WorkerPoolManager,
        )

        session = next(get_db())
        try:
            video_repo = SqlVideoRepository(session)
            task_repo = SQLAlchemyTaskRepository(session)
            orchestrator = TaskOrchestrator(task_repo, video_repo)

            pool_manager = WorkerPoolManager(orchestrator)
            hash_config = WorkerConfig(TaskType.HASH, 1, ResourceType.CPU, 1)
            pool_manager.add_worker_pool(hash_config)

            # Actually start the worker pools
            pool_manager.start_all()

            print("✅ Worker pool started successfully")
            return {"status": "success", "message": "Worker pools started and running"}

        finally:
            session.close()

    except Exception as e:
        print(f"❌ Worker pool test failed: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}
