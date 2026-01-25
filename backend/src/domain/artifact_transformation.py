"""Artifact envelope transformation utilities.

This module provides functions to transform ML Service responses into
ArtifactEnvelopes for persistence to PostgreSQL.

The transformation process:
1. Extract individual detections/segments from batch ML response
2. Copy provenance metadata (config_hash, input_hash, run_id, etc.)
3. Create ArtifactEnvelope for each detection/segment
4. Return list of envelopes for batch insertion
"""

import json
import logging
from datetime import datetime

from .artifacts import ArtifactEnvelope

logger = logging.getLogger(__name__)


def transform_to_envelopes(
    ml_response: dict,
    task_id: str,
    video_id: str,
    task_type: str,
    config_hash: str,
    input_hash: str,
    run_id: str,
    producer: str = "ml-service",
    producer_version: str = "1.0.0",
    model_profile: str = "balanced",
) -> list[ArtifactEnvelope]:
    """Transform ML Service response to ArtifactEnvelopes.

    This function extracts individual detections/segments from a batch ML
    response and creates ArtifactEnvelopes with complete provenance metadata.

    Args:
        ml_response: ML Service response dict containing detections/segments
        task_id: Task identifier (for logging)
        video_id: Video identifier (asset_id)
        task_type: Type of task (e.g., 'object_detection', 'transcription')
        config_hash: Hash of configuration used for inference
        input_hash: Hash of input data (video)
        run_id: Run identifier for linking to execution context
        producer: Producer name (default: 'ml-service')
        producer_version: Producer version (default: '1.0.0')
        model_profile: Model profile used (default: 'balanced')

    Returns:
        List of ArtifactEnvelope objects ready for batch insertion

    Raises:
        ValueError: If ml_response is missing required fields
        KeyError: If ml_response structure is invalid
    """
    if not ml_response:
        raise ValueError("ml_response cannot be empty")

    envelopes = []

    # Map task types to artifact types and response field names
    task_to_artifact_type = {
        "object_detection": "object_detection",
        "face_detection": "face_detection",
        "transcription": "transcript_segment",
        "ocr": "ocr_detection",
        "place_detection": "place_classification",
        "scene_detection": "scene",
    }

    artifact_type = task_to_artifact_type.get(task_type)
    if not artifact_type:
        raise ValueError(f"Unknown task type: {task_type}")

    # Extract detections/segments from response
    # Response structure varies by task type
    detections = ml_response.get("detections", [])
    if not detections:
        logger.warning(
            f"No detections found in ML response for task {task_id} ({task_type})"
        )
        return []

    # Transform each detection to an ArtifactEnvelope
    for idx, detection in enumerate(detections):
        try:
            # Generate unique artifact ID
            artifact_id = f"{video_id}_{task_type}_{run_id}_{idx}"

            # Extract time span (in milliseconds)
            span_start_ms = int(detection.get("start_ms", 0))
            span_end_ms = int(detection.get("end_ms", 0))

            # Validate time span
            if span_start_ms < 0 or span_end_ms < 0:
                logger.warning(
                    f"Invalid time span for detection {idx}: "
                    f"start={span_start_ms}, end={span_end_ms}"
                )
                continue

            if span_start_ms > span_end_ms:
                logger.warning(
                    f"Invalid time span for detection {idx}: "
                    f"start > end ({span_start_ms} > {span_end_ms})"
                )
                continue

            # Serialize detection payload to JSON
            payload_json = json.dumps(detection)

            # Create ArtifactEnvelope
            envelope = ArtifactEnvelope(
                artifact_id=artifact_id,
                asset_id=video_id,
                artifact_type=artifact_type,
                schema_version=1,  # Current schema version
                span_start_ms=span_start_ms,
                span_end_ms=span_end_ms,
                payload_json=payload_json,
                producer=producer,
                producer_version=producer_version,
                model_profile=model_profile,
                config_hash=config_hash,
                input_hash=input_hash,
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            envelopes.append(envelope)
            logger.debug(f"Created artifact envelope {artifact_id} for task {task_id}")

        except (ValueError, KeyError) as e:
            logger.error(f"Error transforming detection {idx} for task {task_id}: {e}")
            continue

    logger.info(
        f"Transformed {len(envelopes)} detections to ArtifactEnvelopes "
        f"for task {task_id} ({task_type})"
    )
    return envelopes


def validate_ml_response(ml_response: dict, task_type: str) -> bool:
    """Validate ML Service response structure.

    Args:
        ml_response: ML Service response to validate
        task_type: Type of task (for validation rules)

    Returns:
        True if response is valid, False otherwise
    """
    if not isinstance(ml_response, dict):
        logger.error(f"ML response must be dict, got {type(ml_response)}")
        return False

    if "detections" not in ml_response:
        logger.error("ML response missing 'detections' field")
        return False

    detections = ml_response["detections"]
    if not isinstance(detections, list):
        logger.error(f"'detections' must be list, got {type(detections)}")
        return False

    # Validate each detection has required fields
    for idx, detection in enumerate(detections):
        if not isinstance(detection, dict):
            logger.error(f"Detection {idx} must be dict, got {type(detection)}")
            return False

        if "start_ms" not in detection or "end_ms" not in detection:
            logger.error(f"Detection {idx} missing 'start_ms' or 'end_ms' field")
            return False

    return True
