"""Redis result polling for Worker Service.

This module handles polling Redis for ML Service results with exponential backoff
and automatic cleanup.
"""

import asyncio
import json
import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Polling configuration
POLL_INITIAL_DELAY = 1.0  # Start with 1 second
POLL_MAX_DELAY = 30.0  # Cap at 30 seconds
POLL_TIMEOUT = 1800.0  # 30 minutes total timeout


class RedisResultPoller:
    """Polls Redis for ML Service results with exponential backoff."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the poller with Redis connection details.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client = None

    async def connect(self):
        """Establish Redis connection."""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.debug("Connected to Redis")

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.debug("Closed Redis connection")

    async def poll_for_result(
        self,
        task_id: str,
        initial_delay: float = POLL_INITIAL_DELAY,
        max_delay: float = POLL_MAX_DELAY,
        timeout: float = POLL_TIMEOUT,
    ) -> dict:
        """Poll Redis for ML Service result with exponential backoff.

        Polls the Redis key "ml_result:{task_id}" until the result is available.
        Uses exponential backoff to avoid excessive Redis queries.

        Args:
            task_id: Task identifier
            initial_delay: Initial polling delay in seconds (default: 1.0)
            max_delay: Maximum polling delay in seconds (default: 30.0)
            timeout: Total polling timeout in seconds (default: 1800.0)

        Returns:
            ML Service result dict

        Raises:
            TimeoutError: If polling timeout is exceeded
            ValueError: If result key expires before retrieval
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected. Call connect() first.")

        result_key = f"ml_result:{task_id}"
        elapsed = 0.0
        delay = initial_delay

        logger.info(
            f"Starting Redis result polling for task {task_id} " f"(timeout={timeout}s)"
        )

        while elapsed < timeout:
            try:
                # Try to get result from Redis
                result_json = await self.redis_client.get(result_key)

                if result_json:
                    # Result found, deserialize and return
                    result = json.loads(result_json)
                    logger.info(f"ML result found for task {task_id}")
                    return result

                # Result not yet available, wait and retry
                logger.debug(
                    f"Result not yet available for task {task_id}, "
                    f"waiting {delay}s before retry"
                )
                await asyncio.sleep(delay)
                elapsed += delay
                delay = min(delay * 2, max_delay)  # Exponential backoff capped

            except json.JSONDecodeError as e:
                logger.error(f"Failed to deserialize result for task {task_id}: {e}")
                raise ValueError(
                    f"Invalid JSON in Redis result for task {task_id}"
                ) from e

            except Exception as e:
                logger.error(f"Error polling Redis for task {task_id}: {e}")
                # Continue polling on error
                await asyncio.sleep(delay)
                elapsed += delay
                delay = min(delay * 2, max_delay)

        # Timeout exceeded
        raise TimeoutError(
            f"Redis result polling timeout exceeded for task {task_id} "
            f"after {timeout}s"
        )

    async def delete_result(self, task_id: str) -> bool:
        """Delete result from Redis after successful processing.

        Args:
            task_id: Task identifier

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            RuntimeError: If Redis client not connected
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected. Call connect() first.")

        result_key = f"ml_result:{task_id}"

        try:
            deleted = await self.redis_client.delete(result_key)
            if deleted:
                logger.info(f"Deleted Redis result key for task {task_id}")
            else:
                logger.warning(f"Result key not found for task {task_id}")
            return bool(deleted)

        except Exception as e:
            logger.error(f"Error deleting Redis result for task {task_id}: {e}")
            raise

    async def check_result_exists(self, task_id: str) -> bool:
        """Check if result exists in Redis without retrieving it.

        Args:
            task_id: Task identifier

        Returns:
            True if result exists, False otherwise

        Raises:
            RuntimeError: If Redis client not connected
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected. Call connect() first.")

        result_key = f"ml_result:{task_id}"

        try:
            exists = await self.redis_client.exists(result_key)
            return bool(exists)

        except Exception as e:
            logger.error(f"Error checking Redis result for task {task_id}: {e}")
            raise
