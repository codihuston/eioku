"""Hashing utilities for provenance tracking."""

import hashlib
import json
from pathlib import Path


def compute_config_hash(config: dict) -> str:
    """Compute hash of configuration for provenance tracking.

    Args:
        config: Configuration dictionary

    Returns:
        Hex digest of configuration hash (first 16 chars)
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_input_hash(video_path: str) -> str:
    """Compute hash of input video file for provenance tracking.

    Args:
        video_path: Path to video file

    Returns:
        Hex digest of file hash (first 16 chars)
    """
    path = Path(video_path)

    if not path.exists():
        # If file doesn't exist, use path as identifier
        return hashlib.sha256(video_path.encode()).hexdigest()[:16]

    # Compute file hash
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()[:16]
