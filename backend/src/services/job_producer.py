"""Job producer for enqueueing ML tasks to Redis via arq."""

import logging

from arq import create_pool

from ..config.redis_config import get_redis_url

logger = logging.getLogger(__name__)


class JobProducer:
    """Produces jobs for ML task execution.

    All jobs are enqueued to a single 'jobs' queue. Workers consume from this queue
    and request the next job they can handle based on their capabilities (GPU_MODE).
    """

    # Task types that require GPU
    GPU_REQUIRED_TASKS = {
        "object_detection",
        "face_detection",
        "place_detection",
        "scene_detection",
    }

    # Task types that can run on CPU or GPU
    CPU_CAPABLE_TASKS = {
        "transcription",
        "ocr",
    }

    # All supported tasks
    SUPPORTED_TASKS = GPU_REQUIRED_TASKS | CPU_CAPABLE_TASKS

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

    def _get_queue_name(self, task_type: str) -> str:
        """Determine if task type is supported.

        All tasks go to the single 'jobs' queue. Workers will filter based on
        their capabilities (GPU_MODE).

        Args:
            task_type: Type of task (e.g., 'object_detection', 'transcription')

        Returns:
            Queue name: 'jobs' (single queue for all tasks)

        Raises:
            ValueError: If task_type is not recognized
        """
        if task_type not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unknown task type: {task_type}")

        # All tasks go to the single 'jobs' queue
        return "jobs"

    def can_worker_handle(self, task_type: str, gpu_available: bool) -> bool:
        """Check if a worker can handle a task based on its capabilities.

        Args:
            task_type: Type of task
            gpu_available: Whether the worker has GPU available

        Returns:
            True if worker can handle this task type
        """
        if task_type not in self.SUPPORTED_TASKS:
            return False

        # GPU-required tasks need GPU
        if task_type in self.GPU_REQUIRED_TASKS:
            return gpu_available

        # CPU-capable tasks can run on any worker
        if task_type in self.CPU_CAPABLE_TASKS:
            return True

        return False

    async def enqueue_task(
        self,
        task_id: str,
        task_type: str,
        video_id: str,
        video_path: str,
        config: dict | None = None,
    ) -> str:
        """Enqueue a task to the single jobs queue.

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

        if task_type not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unknown task type: {task_type}")

        queue_name = "jobs"  # Single queue for all tasks
        job_id = f"ml_{task_id}"

        # Prepare job payload
        job_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "video_id": video_id,
            "video_path": video_path,
            "config": config or {},
        }

        # Enqueue job to the single jobs queue
        # arq uses XADD to add to stream, with job_id for deduplication
        await self.pool.enqueue_job(
            "process_ml_task",
            job_id=job_id,
            _queue_name=queue_name,
            **job_payload,
        )

        logger.info(
            f"Enqueued task {task_id} ({task_type}) to {queue_name} "
            f"queue with job_id {job_id}"
        )
        return job_id

    async def enqueue_to_ml_jobs(
        self,
        task_id: str,
        task_type: str,
        video_id: str,
        video_path: str,
        config: dict | None = None,
    ) -> str:
        """Enqueue a job to the ml_jobs queue for ML Service processing.

        This is called by the Worker Service after consuming a job from
        gpu_jobs/cpu_jobs. The job is enqueued to ml_jobs for the ML Service
        to process.

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

        if task_type not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unknown task type: {task_type}")

        queue_name = "ml_jobs"
        job_id = f"ml_{task_id}"

        # Prepare job payload for ML Service
        job_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "video_id": video_id,
            "video_path": video_path,
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
