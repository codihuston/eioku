"""Job producer for enqueueing ML tasks to Redis via arq."""

import logging

from arq import create_pool

from ..config.redis_config import get_redis_url

logger = logging.getLogger(__name__)


class JobProducer:
    """Produces jobs for ML task execution.

    All jobs are enqueued to the ml_jobs queue for the ml-service worker to consume.
    The ml-service worker then processes jobs and pushes results to Redis.
    """

    # All supported tasks
    SUPPORTED_TASKS = {
        "object_detection",
        "face_detection",
        "transcription",
        "ocr",
        "place_detection",
        "scene_detection",
        "metadata_extraction",
    }

    def __init__(self, redis_url: str | None = None):
        """Initialize JobProducer with Redis connection.

        Args:
            redis_url: Redis connection URL (default: from redis_config.py)
        """
        self.redis_url = redis_url or get_redis_url()
        self.pool = None

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        from ..config.redis_config import get_redis_settings

        redis_settings = get_redis_settings()
        self.pool = await create_pool(redis_settings)
        logger.info(f"JobProducer initialized with Redis: {self.redis_url}")

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("JobProducer connection closed")

    def _validate_task_type(self, task_type: str) -> None:
        """Validate that task type is supported.

        Args:
            task_type: Type of task (e.g., 'object_detection', 'transcription')

        Raises:
            ValueError: If task_type is not recognized
        """
        if task_type not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unknown task type: {task_type}")

    async def enqueue_task(
        self,
        task_id: str,
        task_type: str,
        video_id: str,
        video_path: str,
        config: dict | None = None,
    ) -> str:
        """Enqueue a job to the ml_jobs queue for ml-service worker to consume.

        The ml-service worker will dequeue this job and process it.

        Args:
            task_id: Unique task identifier
            task_type: Type of task (e.g., 'object_detection')
            video_id: Video identifier
            video_path: Path to video file
            config: Optional task configuration

        Returns:
            Job ID in format "ml_{task_id}"

        Raises:
            ValueError: If task_type is not recognized
            RuntimeError: If Redis connection not initialized
        """
        if not self.pool:
            raise RuntimeError("JobProducer not initialized. Call initialize() first.")

        self._validate_task_type(task_type)

        queue_name = "ml_jobs"
        job_id = f"ml_{task_id}"

        # Enqueue job to ml_jobs queue using arq's zadd
        # arq stores jobs in a sorted set with the queue name as the key
        # The _job_id ensures deduplication if the same task is enqueued twice
        job = await self.pool.enqueue_job(
            "process_ml_task",
            task_id,
            task_type,
            video_id,
            video_path,
            config or {},
            _job_id=job_id,
            _queue_name=queue_name,
        )

        if job is None:
            logger.warning(
                f"Job {job_id} already exists in Redis (duplicate enqueue attempt)"
            )
            return job_id

        logger.info(
            f"Enqueued task {task_id} ({task_type}) to {queue_name} queue "
            f"with job_id {job_id}"
        )

        # Verify job was actually written to Redis
        jobs_in_queue = await self.pool.zrange(queue_name, 0, -1)
        if job_id.encode() not in jobs_in_queue:
            logger.error(
                f"Job {job_id} was not found in Redis queue {queue_name} "
                f"after enqueueing. Jobs in queue: {jobs_in_queue}"
            )
        else:
            logger.debug(f"Verified job {job_id} is in Redis queue {queue_name}")

        return job_id
