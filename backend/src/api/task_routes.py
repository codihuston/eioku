"""Task processing and status API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..repositories.task_repository import SQLAlchemyTaskRepository
from ..repositories.video_repository import SqlVideoRepository
from ..services.task_orchestrator import TaskOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


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
            import src.main as main_module

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
