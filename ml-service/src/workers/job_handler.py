"""Job handler for ML Service job consumption from ml_jobs queue.

This module implements the process_inference_job() handler that:
1. Reads job payload (task_id, task_type, video_id, video_path, config)
2. Executes appropriate ML inference (object detection, face detection, etc.)
3. Publishes results to Redis key for Worker Service to consume
4. Acknowledges job in Redis (XACK) on successful completion
5. Does NOT acknowledge on failure (allows arq to retry)
"""

import json
import logging

import redis.asyncio as redis

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
    It executes ML inference and publishes results to Redis for Worker Service.

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
        Dictionary with task_id, status, and result_key

    Raises:
        ValueError: If task_type is not recognized
        RuntimeError: If inference fails
        Exception: If Redis operations fail (job will be retried by arq)
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

    redis_client = None
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

        # Step 2: Publish results to Redis for Worker Service
        import os

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = await redis.from_url(redis_url)

        result_key = f"ml_result:{task_id}"
        result_json = json.dumps(ml_response)

        # RPUSH to result key (Worker Service will BLPOP)
        await redis_client.rpush(result_key, result_json)
        logger.info(
            f"Published ML results for task {task_id} to Redis key {result_key}"
        )

        return {
            "task_id": task_id,
            "status": "completed",
            "result_key": result_key,
        }

    except Exception as e:
        logger.error(
            f"Error processing inference job for task {task_id}: {e}",
            exc_info=True,
        )
        # Re-raise to allow arq to retry (job will NOT be acknowledged)
        raise

    finally:
        if redis_client:
            await redis_client.close()


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
