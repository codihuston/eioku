"""Job producer for enqueueing ML tasks to Redis via arq."""

import logging

from arq import create_pool

from ..config.redis_config import get_redis_url

logger = logging.getLogger(__name__)


class JobProducer:
    """Produces jobs for ML task execution.

    All jobs are enqueued to the ml_jobs queue for the ML Service to process.
    The ML Service handles all task types regardless of GPU availability.
    """

    # All supported tasks
    SUPPORTED_TASKS = {
        "object_detection",
        "face_detection",
        "transcription",
        "ocr",
        "place_detection",
        "scene_detection",
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

    async def enqueue_to_ml_jobs(
        self,
        task_id: str,
        task_type: str,
        video_id: str,
        video_path: str,
        input_hash: str,
        config: dict | None = None,
    ) -> str:
        """Enqueue a job to the ml_jobs queue for ML Service processing.

        The job is enqueued to ml_jobs for the ML Service to process.
        The ML Service handles all task types regardless of GPU availability.

        Args:
            task_id: Unique task identifier
            task_type: Type of task (e.g., 'object_detection')
            video_id: Video identifier
            video_path: Path to video file
            input_hash: xxhash64 of video file (from discovery service)
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

        # Prepare job payload for ML Service
        job_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "video_id": video_id,
            "video_path": video_path,
            "input_hash": input_hash,
            "config": config or {},
        }

        # Enqueue job to ml_jobs queue using arq's XADD
        # The job_id ensures deduplication if the same task is enqueued twice
        await self.pool.enqueue_job(
            "process_inference_job",
            job_id=job_id,
            _queue_name=queue_name,
            **job_payload,
        )

        logger.info(
            f"Enqueued task {task_id} ({task_type}) to {queue_name} "
            f"queue with job_id {job_id}"
        )
        return job_id
