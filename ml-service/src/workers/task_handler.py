"""Task handler for ML Service Worker job consumption and result processing.

This module implements the process_ml_task handler that:
1. Consumes jobs from Redis ml_jobs queue (via arq)
2. Marks task as RUNNING in PostgreSQL
3. Runs ML inference using the model manager
4. Transforms results to ArtifactEnvelopes
5. Batch inserts artifacts to PostgreSQL
6. Marks task as COMPLETED
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)


async def process_ml_task(
    ctx,
    task_id: str,
    task_type: str,
    video_id: str,
    video_path: str,
    config: dict | None = None,
) -> dict:
    """Process an ML task: run inference and persist results to PostgreSQL.

    This handler is called by arq when a job is consumed from the ml_jobs queue.
    It implements the processing pattern:
    1. Mark task as RUNNING in PostgreSQL
    2. Run ML inference using the model manager
    3. Transform results to ArtifactEnvelopes
    4. Batch insert artifacts to PostgreSQL
    5. Mark task as COMPLETED
    6. Sync projections for each artifact

    Args:
        ctx: arq job context
        task_id: Unique task identifier
        task_type: Type of task (e.g., 'object_detection')
        video_id: Video identifier
        video_path: Path to video file
        config: Optional task configuration

    Returns:
        Dictionary with task_id, status, and artifact_count

    Raises:
        asyncio.CancelledError: If task is cancelled via arq
        ValueError: If task parameters are invalid
        RuntimeError: If inference or database operations fail
    """
    session = None
    try:
        logger.info(f"🚀 Dequeued task {task_id} ({task_type}) for video {video_id}")

        # Initialize database session
        from ..database.connection import get_scoped_db, remove_scoped_session

        session = get_scoped_db()
        logger.info(f"✅ Database session initialized for task {task_id}")

        # Mark task as RUNNING
        from ..database.models import Task

        task = session.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found in database")

        task.status = "running"
        task.started_at = datetime.utcnow()
        session.commit()
        logger.info(f"📍 Task {task_id} marked as RUNNING")

        # Initialize model manager for this task
        from src.services.model_manager import ModelManager

        model_cache_dir = os.getenv("MODEL_CACHE_DIR", "/models")
        os.environ["HF_HOME"] = os.path.join(model_cache_dir, "huggingface")
        os.environ["YOLO_HOME"] = os.path.join(model_cache_dir, "ultralytics")
        os.environ["EASYOCR_HOME"] = os.path.join(model_cache_dir, "easyocr")

        model_manager = ModelManager(cache_dir=model_cache_dir)
        logger.info(f"✅ Model manager initialized for task {task_id}")

        # Map task type to inference function
        task_to_endpoint = {
            "object_detection": "objects",
            "face_detection": "faces",
            "transcription": "transcribe",
            "ocr": "ocr",
            "place_detection": "places",
            "scene_detection": "scenes",
        }
        endpoint = task_to_endpoint.get(task_type)
        if not endpoint:
            raise ValueError(f"Unknown task type: {task_type}")

        logger.info(f"🎬 Starting {endpoint} inference on {video_path}")

        # Run inference based on task type
        if task_type == "object_detection":
            result = await model_manager.detect_objects(video_path, config or {})
        elif task_type == "face_detection":
            result = await model_manager.detect_faces(video_path, config or {})
        elif task_type == "transcription":
            result = await model_manager.transcribe_video(video_path, config or {})
        elif task_type == "ocr":
            result = await model_manager.extract_ocr(video_path, config or {})
        elif task_type == "place_detection":
            result = await model_manager.classify_places(video_path, config or {})
        elif task_type == "scene_detection":
            result = await model_manager.detect_scenes(video_path, config or {})
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        logger.info(f"✅ Video processing complete for task {task_id} ({task_type})")

        # Convert result to dict if needed
        if hasattr(result, "dict"):
            result_dict = result.dict()
        else:
            result_dict = result

        logger.info(
            f"📊 Inference results: {len(result_dict.get('detections', []))} "
            f"detections/segments for task {task_id}"
        )

        # Transform results to ArtifactEnvelopes
        from ..domain.artifacts import ArtifactEnvelope

        run_id = str(uuid4())
        envelopes = []

        # Extract metadata from result
        config_hash = result_dict.get("config_hash", "")
        input_hash = result_dict.get("input_hash", "")
        producer = result_dict.get("producer", "ml-service")
        producer_version = result_dict.get("producer_version", "1.0.0")
        model_profile = result_dict.get("model_profile", "balanced")

        # Map task types to artifact types
        task_to_artifact_type = {
            "object_detection": "object.detection",
            "face_detection": "face.detection",
            "transcription": "transcript.segment",
            "ocr": "ocr.text",
            "place_detection": "place.classification",
            "scene_detection": "scene",
        }

        artifact_type = task_to_artifact_type.get(task_type)
        if not artifact_type:
            raise ValueError(f"Unknown task type: {task_type}")

        # Map task types to result keys
        task_to_result_key = {
            "object_detection": "detections",
            "face_detection": "detections",
            "transcription": "segments",
            "ocr": "detections",
            "place_detection": "classifications",
            "scene_detection": "scenes",
        }

        result_key = task_to_result_key.get(task_type)
        if not result_key:
            raise ValueError(f"Unknown task type: {task_type}")

        # Extract detections/segments from response
        detections = result_dict.get(result_key, [])
        if not detections:
            logger.warning(
                f"⚠️  No detections found in inference results for task {task_id}"
            )
        else:
            logger.info(
                f"🔄 Processing {len(detections)} detections into ArtifactEnvelopes "
                f"for task {task_id}"
            )

            # Transform each detection to an ArtifactEnvelope
            for idx, detection in enumerate(detections):
                try:
                    # Generate unique artifact ID
                    artifact_id = f"{video_id}_{task_type}_{run_id}_{idx}"

                    # Extract time span (in milliseconds)
                    # Some detections have explicit start_ms/end_ms
                    # (transcription, scenes)
                    # Others have timestamp_ms (point-in-time detections)
                    if "start_ms" in detection and "end_ms" in detection:
                        span_start_ms = int(detection.get("start_ms", 0))
                        span_end_ms = int(detection.get("end_ms", 0))
                    elif "timestamp_ms" in detection:
                        # For point-in-time detections, use timestamp as
                        # both start and end
                        timestamp_ms = int(detection.get("timestamp_ms", 0))
                        span_start_ms = timestamp_ms
                        span_end_ms = timestamp_ms
                    else:
                        msg = (
                            f"⚠️  No time information in detection {idx} "
                            f"for task {task_id}"
                        )
                        logger.warning(msg)
                        continue

                    # Validate time span
                    if span_start_ms < 0 or span_end_ms < 0:
                        logger.warning(
                            f"⚠️  Invalid time span for detection {idx}: "
                            f"start={span_start_ms}, end={span_end_ms}"
                        )
                        continue

                    if span_start_ms > span_end_ms:
                        logger.warning(
                            f"⚠️  Invalid time span for detection {idx}: "
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
                        schema_version=1,
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

                except (ValueError, KeyError) as e:
                    logger.error(
                        f"❌ Error transforming detection {idx} for task {task_id}: {e}"
                    )
                    continue

            logger.info(
                f"✅ Transformed {len(envelopes)} detections to ArtifactEnvelopes "
                f"for task {task_id}"
            )

        # Batch insert artifacts to PostgreSQL
        if envelopes:
            logger.info(
                f"💾 Batch inserting {len(envelopes)} artifacts to database "
                f"for task {task_id} and video {video_id}"
            )

            from ..database.models import Artifact

            # Convert ArtifactEnvelope domain objects to ORM models
            orm_envelopes = []
            for envelope in envelopes:
                orm_envelope = Artifact(
                    artifact_id=envelope.artifact_id,
                    asset_id=envelope.asset_id,
                    artifact_type=envelope.artifact_type,
                    schema_version=envelope.schema_version,
                    span_start_ms=envelope.span_start_ms,
                    span_end_ms=envelope.span_end_ms,
                    payload_json=envelope.payload_json,
                    producer=envelope.producer,
                    producer_version=envelope.producer_version,
                    model_profile=envelope.model_profile,
                    config_hash=envelope.config_hash,
                    input_hash=envelope.input_hash,
                    run_id=envelope.run_id,
                    created_at=envelope.created_at,
                )
                orm_envelopes.append(orm_envelope)

            # Batch insert
            session.add_all(orm_envelopes)
            session.commit()
            logger.info(
                f"✅ Successfully inserted {len(orm_envelopes)} artifacts to "
                f"PostgreSQL for task {task_id}"
            )

            # Sync projections for each artifact
            logger.info(
                f"🔄 Syncing projections for {len(envelopes)} artifacts "
                f"for task {task_id}"
            )

            from ..services.projection_sync_service import ProjectionSyncService

            projection_service = ProjectionSyncService(session)
            for envelope in envelopes:
                try:
                    projection_service.sync_artifact(envelope)
                except Exception as e:
                    logger.warning(
                        f"⚠️  Failed to sync projection for artifact "
                        f"{envelope.artifact_id}: {e}"
                    )

            logger.info(
                f"✅ Projection sync complete for task {task_id} "
                f"({len(envelopes)} artifacts)"
            )

        # Mark task as COMPLETED
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        session.commit()
        logger.info(
            f"✅ Task {task_id} ({task_type}) marked as COMPLETED "
            f"({len(envelopes)} artifacts persisted)"
        )

        return {
            "task_id": task_id,
            "status": "completed",
            "artifact_count": len(envelopes),
        }

    except asyncio.CancelledError:
        logger.warning(f"⚠️  Task {task_id} was cancelled via arq")
        if session:
            try:
                from ..database.models import Task

                task = session.query(Task).filter(Task.task_id == task_id).first()
                if task:
                    task.status = "cancelled"
                    session.commit()
                    logger.info(f"📍 Task {task_id} marked as CANCELLED")
            except Exception as e:
                logger.error(f"❌ Failed to mark task as cancelled: {e}")
        raise

    except Exception as e:
        logger.error(f"❌ Error processing task {task_id}: {e}", exc_info=True)

        # Mark task as FAILED
        if session:
            try:
                from ..database.models import Task

                task = session.query(Task).filter(Task.task_id == task_id).first()
                if task:
                    task.status = "failed"
                    task.error_message = str(e)
                    session.commit()
                    logger.info(f"📍 Task {task_id} marked as FAILED")
            except Exception as db_error:
                logger.error(f"❌ Failed to mark task as failed: {db_error}")

        raise RuntimeError(f"Failed to process task {task_id}: {e}")

    finally:
        # Close database session
        if session:
            session.close()
            logger.debug(f"Database session closed for task {task_id}")

        # Remove scoped session
        from ..database.connection import remove_scoped_session

        remove_scoped_session()
        logger.debug(f"Scoped session removed for task {task_id}")
