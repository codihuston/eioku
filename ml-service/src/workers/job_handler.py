"""Job handler for ML Service job consumption from ml_jobs queue.

This module implements the process_inference_job() handler that:
1. Reads job payload (task_id, task_type, video_id, video_path, config)
2. Executes appropriate ML inference (object detection, face detection, etc.)
3. Creates ArtifactEnvelopes with provenance metadata
4. Batch inserts artifacts to PostgreSQL in single transaction
5. Acknowledges job in Redis (XACK) on successful completion
6. Does NOT acknowledge on failure (allows arq to retry)
"""

import json
import logging

logger = logging.getLogger(__name__)

# Task type to inference endpoint mapping
TASK_TYPE_TO_ENDPOINT = {
    "object_detection": "objects",
    "face_detection": "faces",
    "transcription": "transcribe",
    "ocr": "ocr",
    "place_detection": "places",
    "scene_detection": "scenes",
}


async def process_inference_job(
    task_id: str,
    task_type: str,
    video_id: str,
    video_path: str,
    input_hash: str,
    config: dict | None = None,
) -> dict:
    """Process an ML inference job from the ml_jobs queue.

    This handler is called by arq when a job is consumed from the ml_jobs queue.
    It executes ML inference and persists results to PostgreSQL.

    The job payload is expected to contain:
    - task_id: Unique task identifier
    - task_type: Type of task (e.g., 'object_detection')
    - video_id: Video identifier
    - video_path: Path to video file
    - input_hash: xxhash64 of video file (from discovery service)
    - config: Task configuration (optional)

    Args:
        task_id: Unique task identifier
        task_type: Type of task (e.g., 'object_detection')
        video_id: Video identifier
        video_path: Path to video file
        input_hash: xxhash64 of video file
        config: Optional task configuration

    Returns:
        Dictionary with task_id, status, and artifact_count

    Raises:
        ValueError: If task_type is not recognized
        RuntimeError: If inference fails
        Exception: If database operations fail (job will be retried by arq)
    """
    if not config:
        config = {}

    logger.info(
        f"Processing inference job: task_id={task_id}, task_type={task_type}, "
        f"video_id={video_id}, video_path={video_path}"
    )

    # Validate task type
    if task_type not in TASK_TYPE_TO_ENDPOINT:
        raise ValueError(f"Unknown task type: {task_type}")

    try:
        # Step 1: Execute ML inference
        logger.info(f"Executing {task_type} inference for task {task_id}")
        ml_response = await _execute_inference(
            task_type=task_type,
            video_path=video_path,
            input_hash=input_hash,
            config=config,
        )

        logger.info(
            f"Inference completed for task {task_id}: "
            f"got {len(ml_response.get('detections', []))} detections"
        )

        # Step 2: Transform ML response to ArtifactEnvelopes
        logger.info(f"Transforming artifacts for task {task_id}")
        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
        )

        logger.info(f"Transformed {len(artifacts)} artifacts for task {task_id}")

        # Step 3: Batch insert artifacts to PostgreSQL
        logger.info(f"Persisting {len(artifacts)} artifacts for task {task_id}")
        artifact_count = await _persist_artifacts(
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
            artifacts=artifacts,
        )

        logger.info(
            f"Successfully persisted {artifact_count} artifacts for task {task_id}"
        )

        # Step 4: Return success (arq will acknowledge the job)
        return {
            "task_id": task_id,
            "status": "completed",
            "artifact_count": artifact_count,
        }

    except Exception as e:
        logger.error(
            f"Error processing inference job for task {task_id}: {e}",
            exc_info=True,
        )
        # Re-raise to allow arq to retry (job will NOT be acknowledged)
        raise


async def _execute_inference(
    task_type: str,
    video_path: str,
    input_hash: str,
    config: dict,
) -> dict:
    """Execute ML inference for the given task type.

    This function calls the appropriate ML inference endpoint based on task_type.
    In a real implementation, this would call the inference endpoints directly
    or via HTTP. For now, it's a placeholder that would be implemented by
    importing the actual inference functions.

    Args:
        task_type: Type of task (e.g., 'object_detection')
        video_path: Path to video file
        input_hash: xxhash64 of video file
        config: Task configuration

    Returns:
        ML response dictionary with detections/segments and provenance metadata

    Raises:
        ValueError: If task_type is not recognized
        RuntimeError: If inference fails
    """
    # Import inference functions from api module
    from ..api import inference

    endpoint = TASK_TYPE_TO_ENDPOINT.get(task_type)
    if not endpoint:
        raise ValueError(f"Unknown task type: {task_type}")

    logger.debug(f"Calling inference endpoint: {endpoint}")

    # Map task types to request models and inference functions
    if task_type == "object_detection":
        from ..models.requests import ObjectDetectionRequest

        request = ObjectDetectionRequest(
            video_path=video_path,
            input_hash=input_hash,
            model_name=config.get("model_name", "yolov8n.pt"),
            frame_interval=config.get("frame_interval", 30),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            model_profile=config.get("model_profile", "balanced"),
        )
        response = await inference.detect_objects(request)

    elif task_type == "face_detection":
        from ..models.requests import FaceDetectionRequest

        request = FaceDetectionRequest(
            video_path=video_path,
            input_hash=input_hash,
            model_name=config.get("model_name", "yolov8n-face.pt"),
            frame_interval=config.get("frame_interval", 30),
            confidence_threshold=config.get("confidence_threshold", 0.5),
        )
        response = await inference.detect_faces(request)

    elif task_type == "transcription":
        from ..models.requests import TranscriptionRequest

        request = TranscriptionRequest(
            video_path=video_path,
            input_hash=input_hash,
            model_name=config.get("model_name", "large-v3"),
            language=config.get("language"),
            vad_filter=config.get("vad_filter", True),
        )
        response = await inference.transcribe_video(request)

    elif task_type == "ocr":
        from ..models.requests import OCRRequest

        request = OCRRequest(
            video_path=video_path,
            input_hash=input_hash,
            frame_interval=config.get("frame_interval", 60),
            languages=config.get("languages", ["en"]),
            use_gpu=config.get("use_gpu", True),
        )
        response = await inference.extract_ocr(request)

    elif task_type == "place_detection":
        from ..models.requests import PlaceDetectionRequest

        request = PlaceDetectionRequest(
            video_path=video_path,
            input_hash=input_hash,
            frame_interval=config.get("frame_interval", 60),
            top_k=config.get("top_k", 5),
        )
        response = await inference.classify_places(request)

    elif task_type == "scene_detection":
        from ..models.requests import SceneDetectionRequest

        request = SceneDetectionRequest(
            video_path=video_path,
            input_hash=input_hash,
            threshold=config.get("threshold", 0.4),
            min_scene_length=config.get("min_scene_length", 0.6),
        )
        response = await inference.detect_scenes(request)

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Convert response to dict for transformation
    return response.model_dump()


def _transform_to_artifacts(
    ml_response: dict,
    task_id: str,
    video_id: str,
    task_type: str,
) -> list[dict]:
    """Transform ML response to artifact dictionaries.

    This function extracts individual detections/segments from the ML response
    and creates artifact dictionaries with provenance metadata.

    Args:
        ml_response: ML Service response dictionary
        task_id: Task identifier (for logging)
        video_id: Video identifier (asset_id)
        task_type: Type of task

    Returns:
        List of artifact dictionaries ready for database insertion

    Raises:
        ValueError: If ml_response is invalid
    """
    if not ml_response:
        raise ValueError("ml_response cannot be empty")

    artifacts = []

    # Extract detections/segments from response
    detections = ml_response.get("detections", [])
    if not detections and task_type != "scene_detection":
        logger.warning(
            f"No detections found in ML response for task {task_id} ({task_type})"
        )
        return []

    # Handle scene detection separately (uses 'scenes' field)
    if task_type == "scene_detection":
        detections = ml_response.get("scenes", [])

    # Transform each detection to an artifact
    for idx, detection in enumerate(detections):
        try:
            # Extract time span (in milliseconds)
            if task_type == "scene_detection":
                span_start_ms = int(detection.get("start_ms", 0))
                span_end_ms = int(detection.get("end_ms", 0))
            else:
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

            # Create artifact dictionary
            artifact = {
                "task_id": task_id,
                "video_id": video_id,
                "task_type": task_type,
                "span_start_ms": span_start_ms,
                "span_end_ms": span_end_ms,
                "payload": detection,
                "config_hash": ml_response.get("config_hash", ""),
                "input_hash": ml_response.get("input_hash", ""),
                "run_id": ml_response.get("run_id", ""),
                "producer": ml_response.get("producer", "ml-service"),
                "producer_version": ml_response.get("producer_version", "1.0.0"),
                "model_profile": ml_response.get("model_profile", "balanced"),
            }

            artifacts.append(artifact)
            logger.debug(f"Created artifact {idx} for task {task_id}")

        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error transforming detection {idx} for task {task_id}: {e}")
            continue

    logger.info(
        f"Transformed {len(artifacts)} detections to artifacts "
        f"for task {task_id} ({task_type})"
    )
    return artifacts


async def _persist_artifacts(
    task_id: str,
    video_id: str,
    task_type: str,
    artifacts: list[dict],
) -> int:
    """Batch insert artifacts to PostgreSQL.

    This function inserts all artifacts for a task in a single transaction.
    If any insert fails, the entire transaction is rolled back.

    Args:
        task_id: Task identifier (for logging)
        video_id: Video identifier
        task_type: Type of task
        artifacts: List of artifact dictionaries

    Returns:
        Number of artifacts inserted

    Raises:
        RuntimeError: If database operations fail
    """
    if not artifacts:
        logger.info(f"No artifacts to persist for task {task_id}")
        return 0

    try:
        # Import database dependencies
        import os
        import re
        from datetime import datetime

        import psycopg2

        # Get database connection string
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://eioku:eioku_dev@localhost:5432/eioku",
        )

        # Parse connection string
        # Format: postgresql://user:password@host:port/database
        match = re.match(
            r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)",
            db_url,
        )
        if not match:
            raise ValueError(f"Invalid DATABASE_URL format: {db_url}")

        user, password, host, port, database = match.groups()

        # Connect to database
        conn = psycopg2.connect(
            host=host,
            port=int(port),
            database=database,
            user=user,
            password=password,
        )

        try:
            cursor = conn.cursor()

            # Begin transaction
            cursor.execute("BEGIN")

            # Insert each artifact
            for artifact in artifacts:
                cursor.execute(
                    """
                    INSERT INTO artifacts (
                        task_id, asset_id, artifact_type,
                        span_start_ms, span_end_ms, payload_json,
                        config_hash, input_hash, run_id,
                        producer, producer_version, model_profile,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        artifact["task_id"],
                        artifact["video_id"],
                        artifact["task_type"],
                        artifact["span_start_ms"],
                        artifact["span_end_ms"],
                        json.dumps(artifact["payload"]),
                        artifact["config_hash"],
                        artifact["input_hash"],
                        artifact["run_id"],
                        artifact["producer"],
                        artifact["producer_version"],
                        artifact["model_profile"],
                        datetime.utcnow(),
                    ),
                )

            # Commit transaction
            conn.commit()
            logger.info(
                f"Successfully inserted {len(artifacts)} artifacts for task {task_id}"
            )

            return len(artifacts)

        finally:
            cursor.close()
            conn.close()

    except Exception as e:
        logger.error(
            f"Error persisting artifacts for task {task_id}: {e}",
            exc_info=True,
        )
        # Re-raise to allow arq to retry
        raise RuntimeError(f"Failed to persist artifacts for task {task_id}: {e}")
