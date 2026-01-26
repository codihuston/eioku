"""Hashing utilities for provenance tracking."""

import json
import logging
from pathlib import Path

import xxhash

logger = logging.getLogger(__name__)


def compute_config_hash(config: dict) -> str:
    """Compute hash of configuration for provenance tracking.

    Args:
        config: Configuration dictionary

    Returns:
        Hex digest of configuration hash (first 16 chars)
    """
    config_str = json.dumps(config, sort_keys=True)
    hasher = xxhash.xxh64(config_str.encode())
    return hasher.hexdigest()[:16]


def compute_input_hash(video_path: str) -> str:
    """Compute xxhash64 of input video file for provenance tracking.

    Uses xxhash64 for fast, consistent hashing. This matches the hashing
    algorithm used by the backend's FileHashService.

    Args:
        video_path: Path to video file

    Returns:
        Hex digest of file hash (first 16 chars)
    """
    path = Path(video_path)

    if not path.exists():
        # If file doesn't exist, use path as identifier
        hasher = xxhash.xxh64(video_path.encode())
        return hasher.hexdigest()[:16]

    # Compute file hash using xxhash64
    hasher = xxhash.xxh64()
    try:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                hasher.update(byte_block)
        return hasher.hexdigest()[:16]
    except Exception as e:
        logger.error(f"Error computing hash for {video_path}: {e}")
        raise


def verify_input_hash(video_path: str, expected_hash: str) -> bool:
    """Verify that a video file's hash matches the expected hash.

    This is a safety check to detect if the video file has changed
    between discovery and processing.

    Args:
        video_path: Path to video file
        expected_hash: Expected hash value (from discovery service)

    Returns:
        True if hashes match, False otherwise

    Raises:
        ValueError: If file doesn't exist or hash computation fails
    """
    path = Path(video_path)

    if not path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    computed_hash = compute_input_hash(video_path)
    matches = computed_hash == expected_hash

    if not matches:
        logger.warning(
            f"Input hash mismatch for {video_path}: "
            f"expected {expected_hash}, got {computed_hash}"
        )

    return matches
