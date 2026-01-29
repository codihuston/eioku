"""Centralized Redis/Valkey connection configuration.

This module provides a single source of truth for Redis connection settings
across all services (API Service, Worker Service, ML Service).

All services should use these settings to ensure consistent Redis connectivity.
"""

import os

from arq.connections import RedisSettings

# Redis/Valkey connection settings
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


def get_redis_url() -> str:
    """Get Redis connection URL.

    Returns:
        Redis URL in format: redis://host:port/db
    """
    return REDIS_URL


def get_redis_settings() -> RedisSettings:
    """Get RedisSettings object for arq.

    Returns:
        RedisSettings configured with host, port, and database
    """
    return REDIS_SETTINGS
