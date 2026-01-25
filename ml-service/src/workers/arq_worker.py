"""arq worker configuration for ML Service.

This module configures the arq worker to consume jobs from the 'ml_jobs'
queue. The ML Service receives jobs enqueued by the Worker Service and
executes ML inference operations.

The ML Service:
- Consumes from the single 'ml_jobs' queue
- Executes ML inference (object detection, face detection, etc.)
- Creates ArtifactEnvelopes with provenance metadata
- Batch inserts artifacts to PostgreSQL
- Acknowledges jobs in Redis on successful completion
"""

import logging
import os

from arq.connections import RedisSettings

logger = logging.getLogger(__name__)

# Redis/Valkey connection settings (centralized)
REDIS_HOST = os.getenv("REDIS_HOST", "valkey")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Build Redis URL for arq
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Create RedisSettings object for arq
REDIS_SETTINGS = RedisSettings(
    host=REDIS_HOST,
    port=REDIS_PORT,
    database=REDIS_DB,
)


class WorkerSettings:
    """arq worker configuration for ML Service.

    This class configures the arq worker with:
    - Single 'ml_jobs' queue for ML inference jobs
    - Redis connection settings
    - Job processing parameters (max_jobs, timeout, retries)
    - Job abort capability for cancellation support
    """

    # Job handler functions (will be populated when handlers are implemented)
    functions = []

    # ML Service consumes from the 'ml_jobs' queue
    queue_name = "ml_jobs"

    # Redis connection settings
    redis_settings = REDIS_SETTINGS

    # Job processing configuration
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", 4))
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", 1800))  # 30 minutes
    max_tries = 3

    # Enable job abort for cancellation support
    allow_abort_jobs = True

    def __init__(self):
        """Initialize worker settings and log configuration."""
        logger.info(
            f"ML Service worker configured: "
            f"queue={self.queue_name}, "
            f"max_jobs={self.max_jobs}, "
            f"job_timeout={self.job_timeout}s, "
            f"redis_host={REDIS_HOST}:{REDIS_PORT}"
        )
