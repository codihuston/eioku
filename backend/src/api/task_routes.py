"""Task processing and status API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..repositories.task_repository import SQLAlchemyTaskRepository
from ..repositories.video_repository import SqlVideoRepository
from ..services.job_producer import JobProducer

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
    description="Enqueue a task that is in PENDING status to the job queue.",
)
async def enqueue_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> EnqueueTaskResponse:
    """Manually enqueue a task for processing."""
    try:
        task_repo = SQLAlchemyTaskRepository(db)
        task = task_repo.find_by_id(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task.status != "pending":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot enqueue task in status {task.status}",
            )

        video_repo = SqlVideoRepository(db)
        video = video_repo.find_by_id(task.video_id)

        if not video:
            raise HTTPException(
                status_code=404,
                detail=f"Video {task.video_id} not found",
            )

        from ..services.video_discovery_service import VideoDiscoveryService

        discovery_service = VideoDiscoveryService(None, video_repo)
        config = discovery_service._get_default_config(task.task_type)

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

            logger.info(f"Enqueued task {task_id} with job_id {job_id}")

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
        raise HTTPException(status_code=500, detail=f"Failed to enqueue task: {str(e)}")


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
    description="Cancel a task that is in PENDING or RUNNING status.",
)
async def cancel_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> CancelTaskResponse:
    """Cancel a task that is in PENDING or RUNNING status."""
    try:
        task_repo = SQLAlchemyTaskRepository(db)
        task = task_repo.find_by_id(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task.status not in ("pending", "running"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task in status {task.status}",
            )

        task.status = "cancelled"
        task_repo.update(task)

        logger.info(f"Task {task_id} marked as CANCELLED")

        return CancelTaskResponse(
            task_id=task_id,
            status="cancelled",
            message="Task cancelled successfully.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


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
    description="Reset a task to PENDING status and re-enqueue it for processing.",
)
async def retry_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> RetryTaskResponse:
    """Retry a failed or cancelled task."""
    try:
        task_repo = SQLAlchemyTaskRepository(db)
        task = task_repo.find_by_id(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task.status not in ("failed", "cancelled"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry task in status {task.status}",
            )

        video_repo = SqlVideoRepository(db)
        video = video_repo.find_by_id(task.video_id)

        if not video:
            raise HTTPException(
                status_code=404,
                detail=f"Video {task.video_id} not found",
            )

        from ..services.video_discovery_service import VideoDiscoveryService

        discovery_service = VideoDiscoveryService(None, video_repo)
        config = discovery_service._get_default_config(task.task_type)

        task.status = "pending"
        task.started_at = None
        task.completed_at = None
        task.error = None
        task_repo.update(task)

        logger.info(f"Task {task_id} reset to PENDING status")

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

            logger.info(f"Retried task {task_id} with job_id {job_id}")

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
        raise HTTPException(status_code=500, detail=f"Failed to retry task: {str(e)}")


@router.post(
    "/reconcile",
    summary="Manually trigger reconciliation",
    description="Run reconciliation to sync pending tasks and re-enqueue missing jobs.",
)
async def manual_reconcile(
    db: Session = Depends(get_db),
) -> dict:
    """Manually trigger reconciliation (one-shot)."""
    from ..services.job_producer import JobProducer
    from ..services.reconciliation_service import ReconciliationService

    try:
        job_producer = JobProducer()
        await job_producer.initialize()
        service = ReconciliationService(db, job_producer)
        stats = await service.run()
        await job_producer.close()
        return {
            "status": "success",
            "message": "Reconciliation completed",
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Manual reconciliation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.get(
    "",
    response_model=TaskListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="List tasks with filtering and sorting",
    description="Get a paginated list of tasks with optional filtering and sorting.",
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
    """List tasks with filtering and sorting."""
    try:
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

        limit = min(limit, 100)
        if limit < 1:
            limit = 10
        if offset < 0:
            offset = 0

        task_repo = SQLAlchemyTaskRepository(db)

        all_tasks = []
        if status:
            all_tasks = task_repo.find_by_status(status)
        else:
            all_tasks = task_repo.find_all() if hasattr(task_repo, "find_all") else []

        if task_type:
            all_tasks = [t for t in all_tasks if t.task_type == task_type]
        if video_id:
            all_tasks = [t for t in all_tasks if str(t.video_id) == video_id]

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

        total = len(all_tasks)
        paginated_tasks = all_tasks[offset : offset + limit]

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

        logger.info(f"Listed {len(paginated_tasks)} tasks (total: {total})")

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
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")
