"""Input validation utilities for inference requests.

This module provides centralized validation for inference requests,
including input hash verification to detect file changes.
"""

import logging
from pathlib import Path

from fastapi import HTTPException

from .hashing import verify_input_hash

logger = logging.getLogger(__name__)


def validate_inference_input(video_path: str, input_hash: str) -> None:
    """Validate inference input by verifying file hash.

    This function checks that:
    1. The video file exists
    2. The file hash matches the expected hash (from discovery service)

    If validation fails, raises HTTPException with appropriate error.

    Args:
        video_path: Path to video file
        input_hash: Expected xxhash64 from discovery service

    Raises:
        HTTPException: If validation fails
            - 400: File not found or hash mismatch
            - 500: Unexpected error during validation
    """
    # Check file exists
    video_file = Path(video_path)
    if not video_file.exists():
        logger.error(f"Video file not found: {video_path}")
        raise HTTPException(
            status_code=400, detail=f"Video file not found: {video_path}"
        )

    if not video_file.is_file():
        logger.error(f"Path is not a file: {video_path}")
        raise HTTPException(status_code=400, detail=f"Path is not a file: {video_path}")

    # Verify hash
    try:
        if not verify_input_hash(video_path, input_hash):
            logger.error(
                f"Input hash verification failed for {video_path}: "
                f"file may have changed since discovery"
            )
            raise HTTPException(
                status_code=400,
                detail="Input hash mismatch: file may have changed since discovery",
            )
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except ValueError as e:
        logger.error(f"Input hash verification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during input validation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal error during input validation"
        )

    logger.debug(f"Input validation passed for {video_path}")
