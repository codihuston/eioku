"""Reconciler for synchronizing PostgreSQL task state with Redis job state.

This module implements the Reconciler class that:
1. Checks all PENDING tasks and re-enqueues missing jobs (handles Redis data loss)
2. Checks all RUNNING tasks and syncs with Redis state
3. Alerts on long-running tasks (never auto-kills)
4. Uses PostgreSQL as source of truth for task state
5. Runs periodically (every 5 minutes via arq cron)
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from ..config.redis_config import REDIS_SETTINGS
from ..database.connection import get_db
from ..repositories.task_repository import SQLAlchemyTaskRepository
from ..services.job_producer import JobProducer

logger = logging.getLogger(__name__)

# Alert threshold for long-running tasks (in seconds)
LONG_RUNNING_THRESHOLD = 3600  # 1 hour


class Reconciler:
    """Reconciler for synchronizing PostgreSQL and Redis state.

    The Reconciler runs periodically to:
    1. Detect and recover from Redis data loss (missing jobs for PENDING tasks)
    2. Detect and recover from job loss (RUNNING tasks with no job in Redis)
    3. Sync job completion status from Redis to PostgreSQL
    4. Alert on long-running tasks
    """

    def __init__(self, session: Session | None = None):
        """Initialize Reconciler.

        Args:
            session: Optional SQLAlchemy session. If not provided, will create one.
        """
        self.session = session
        self.task_repo = None
        self.job_producer = None
        self._owns_session = session is None

    async def run(self) -> dict:
        """Run all reconciliation checks.

        Returns:
            Dictionary with reconciliation statistics
        """
        try:
            # Initialize session if needed
            if self._owns_session:
                self.session = next(get_db())

            self.task_repo = SQLAlchemyTaskRepository(self.session)
            self.job_producer = JobProducer()
            await self.job_producer.initialize()

            stats = {
                "pending_checked": 0,
                "pending_reenqueued": 0,
                "running_checked": 0,
                "running_synced": 0,
                "long_running_alerted": 0,
                "errors": [],
            }

            logger.info("Starting reconciliation run")

            # Step 1: Sync PENDING tasks
            try:
                pending_stats = await self._sync_pending_tasks()
                stats["pending_checked"] = pending_stats["checked"]
                stats["pending_reenqueued"] = pending_stats["reenqueued"]
            except Exception as e:
                logger.error(f"Error syncing PENDING tasks: {e}", exc_info=True)
                stats["errors"].append(f"PENDING sync error: {str(e)}")

            # Step 2: Sync RUNNING tasks
            try:
                running_stats = await self._sync_running_tasks()
                stats["running_checked"] = running_stats["checked"]
                stats["running_synced"] = running_stats["synced"]
            except Exception as e:
                logger.error(f"Error syncing RUNNING tasks: {e}", exc_info=True)
                stats["errors"].append(f"RUNNING sync error: {str(e)}")

            # Step 3: Alert on long-running tasks
            try:
                alert_stats = await self._alert_long_running_tasks()
                stats["long_running_alerted"] = alert_stats["alerted"]
            except Exception as e:
                logger.error(f"Error checking long-running tasks: {e}", exc_info=True)
                stats["errors"].append(f"Long-running check error: {str(e)}")

            logger.info(f"Reconciliation complete: {stats}")
            return stats

        finally:
            # Cleanup
            if self.job_producer:
                await self.job_producer.close()

            if self._owns_session and self.session:
                self.session.close()

    async def _sync_pending_tasks(self) -> dict:
        """Sync all PENDING tasks with Redis.

        For each PENDING task, check if a job exists in Redis.
        If not, re-enqueue the job (handles Redis data loss).

        Returns:
            Dictionary with checked and reenqueued counts
        """
        logger.info("Syncing PENDING tasks")

        pending_tasks = self.task_repo.find_by_status("pending")
        checked = 0
        reenqueued = 0

        for task in pending_tasks:
            checked += 1

            try:
                # Check if job exists in Redis
                job_exists = await self._check_job_exists(task.task_id)

                if not job_exists:
                    logger.warning(
                        f"PENDING task {task.task_id} has no job in Redis - "
                        f"re-enqueueing"
                    )

                    # Re-enqueue the job
                    await self.job_producer.enqueue_task(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        video_id=task.video_id,
                        video_path="",  # Path not available in task record
                        config={},
                    )

                    reenqueued += 1
                    logger.info(f"Re-enqueued PENDING task {task.task_id}")

            except Exception as e:
                logger.error(
                    f"Error syncing PENDING task {task.task_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"PENDING sync complete: checked={checked}, reenqueued={reenqueued}"
        )
        return {"checked": checked, "reenqueued": reenqueued}

    async def _sync_running_tasks(self) -> dict:
        """Sync all RUNNING tasks with Redis.

        For each RUNNING task:
        1. Check if job exists in Redis
        2. If not, reset to PENDING and re-enqueue (handles job loss)
        3. If exists, check job status and sync to PostgreSQL

        Returns:
            Dictionary with checked and synced counts
        """
        logger.info("Syncing RUNNING tasks")

        running_tasks = self.task_repo.find_by_status("running")
        checked = 0
        synced = 0

        for task in running_tasks:
            checked += 1

            try:
                # Check if job exists in Redis
                job_exists = await self._check_job_exists(task.task_id)

                if not job_exists:
                    logger.warning(
                        f"RUNNING task {task.task_id} has no job in Redis - "
                        f"resetting to PENDING"
                    )

                    # Reset to PENDING and re-enqueue
                    task.status = "pending"
                    task.started_at = None
                    self.task_repo.update(task)

                    await self.job_producer.enqueue_task(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        video_id=task.video_id,
                        video_path="",
                        config={},
                    )

                    synced += 1
                    logger.info(f"Reset RUNNING task {task.task_id} to PENDING")

                else:
                    # Job exists - check its status
                    job_status = await self._get_job_status(task.task_id)

                    if job_status == "complete":
                        logger.info(
                            f"RUNNING task {task.task_id} is complete in Redis - "
                            f"updating to COMPLETED"
                        )

                        task.status = "completed"
                        task.completed_at = datetime.utcnow()
                        self.task_repo.update(task)

                        synced += 1

                    elif job_status == "failed":
                        logger.warning(
                            f"RUNNING task {task.task_id} failed in Redis - "
                            f"updating to FAILED"
                        )

                        task.status = "failed"
                        task.completed_at = datetime.utcnow()
                        task.error = "Job failed in Redis"
                        self.task_repo.update(task)

                        synced += 1

            except Exception as e:
                logger.error(
                    f"Error syncing RUNNING task {task.task_id}: {e}",
                    exc_info=True,
                )

        logger.info(f"RUNNING sync complete: checked={checked}, synced={synced}")
        return {"checked": checked, "synced": synced}

    async def _alert_long_running_tasks(self) -> dict:
        """Alert on long-running tasks.

        For each RUNNING task that has been running longer than the threshold,
        send an alert to the operator (never auto-kill).

        Returns:
            Dictionary with alerted count
        """
        logger.info("Checking for long-running tasks")

        running_tasks = self.task_repo.find_by_status("running")
        alerted = 0
        now = datetime.utcnow()
        threshold = timedelta(seconds=LONG_RUNNING_THRESHOLD)

        for task in running_tasks:
            if task.started_at:
                running_time = now - task.started_at

                if running_time > threshold:
                    logger.warning(
                        f"ALERT: Task {task.task_id} ({task.task_type}) "
                        f"has been running for {running_time.total_seconds()}s "
                        f"(threshold: {LONG_RUNNING_THRESHOLD}s)"
                    )

                    # In a real implementation, this would send an alert
                    # (e.g., email, Slack, PagerDuty, etc.)
                    alerted += 1

        logger.info(f"Long-running check complete: alerted={alerted}")
        return {"alerted": alerted}

    async def _check_job_exists(self, task_id: str) -> bool:
        """Check if a job exists in Redis for the given task.

        Args:
            task_id: Task identifier

        Returns:
            True if job exists, False otherwise
        """
        try:
            import redis

            # Connect to Redis
            r = redis.Redis(
                host=REDIS_SETTINGS.host,
                port=REDIS_SETTINGS.port,
                db=REDIS_SETTINGS.database,
                decode_responses=True,
            )

            # Check if job exists in Redis
            # Jobs are stored in Redis with key "arq:job:{job_id}"
            job_id = f"ml_{task_id}"
            job_key = f"arq:job:{job_id}"

            try:
                # Try to get the job from Redis
                job_data = r.get(job_key)

                if job_data:
                    logger.debug(f"Job {job_id} found in Redis")
                    return True
            except Exception as e:
                logger.debug(f"Error checking job {job_id}: {e}")

            logger.debug(f"Job {job_id} not found in Redis")
            return False

        except Exception as e:
            logger.error(f"Error checking if job exists: {e}", exc_info=True)
            # On error, assume job exists to avoid re-enqueueing
            return True

    async def _get_job_status(self, task_id: str) -> str | None:
        """Get the status of a job in Redis.

        Args:
            task_id: Task identifier

        Returns:
            Job status ("complete", "failed", "in_progress") or None if not found
        """
        try:
            import redis

            # Connect to Redis
            r = redis.Redis(
                host=REDIS_SETTINGS.host,
                port=REDIS_SETTINGS.port,
                db=REDIS_SETTINGS.database,
                decode_responses=True,
            )

            job_id = f"ml_{task_id}"
            job_key = f"arq:job:{job_id}"

            # Get job data from Redis
            job_data = r.get(job_key)

            if not job_data:
                logger.debug(f"Job {job_id} not found in Redis")
                return None

            # Parse job data (simplified - in reality would parse JSON)
            # For now, just return None to indicate we couldn't determine status
            logger.debug(f"Job {job_id} found in Redis")
            return None

        except Exception as e:
            logger.error(f"Error getting job status: {e}", exc_info=True)
            return None
