"""Artifact transformation and validation for Worker Service.

This module handles transforming ML Service results from Redis into
ArtifactEnvelopes with schema validation before persistence.
"""

import json
import logging
from typing import Any

from pydantic import ValidationError

from ..domain.schemas import (
    FaceDetectionV1,
    MetadataV1,
    ObjectDetectionV1,
    OCRDetectionV1,
    PlaceClassificationV1,
    SceneV1,
    TranscriptSegmentV1,
)

logger = logging.getLogger(__name__)

# Mapping of task types to artifact types and schema models
ARTIFACT_SCHEMA_MAP = {
    "object_detection": {
        "artifact_type": "object.detection",
        "schema": ObjectDetectionV1,
        "result_key": "detections",
    },
    "face_detection": {
        "artifact_type": "face.detection",
        "schema": FaceDetectionV1,
        "result_key": "detections",
    },
    "transcription": {
        "artifact_type": "transcription",
        "schema": TranscriptSegmentV1,
        "result_key": "segments",
    },
    "ocr": {
        "artifact_type": "ocr",
        "schema": OCRDetectionV1,
        "result_key": "detections",
    },
    "place_detection": {
        "artifact_type": "place.classification",
        "schema": PlaceClassificationV1,
        "result_key": "classifications",
    },
    "scene_detection": {
        "artifact_type": "scene.detection",
        "schema": SceneV1,
        "result_key": "scenes",
    },
    "metadata_extraction": {
        "artifact_type": "video.metadata",
        "schema": MetadataV1,
        "result_key": "metadata",
    },
}


class ArtifactTransformer:
    """Transforms ML Service results into validated ArtifactEnvelopes."""

    @staticmethod
    def transform_ml_result(
        task_id: str,
        task_type: str,
        video_id: str,
        ml_result: dict,
    ) -> list[dict]:
        """Transform ML Service result into ArtifactEnvelopes.

        Extracts individual items from the batch ML result, validates each
        against the schema, and creates ArtifactEnvelopes with provenance.

        Args:
            task_id: Task identifier
            task_type: Type of task (e.g., 'object_detection')
            video_id: Video identifier (asset_id)
            ml_result: ML Service result from Redis (dict)

        Returns:
            List of ArtifactEnvelope dicts ready for persistence

        Raises:
            ValueError: If task_type is unknown or result format is invalid
            ValidationError: If any item fails schema validation
        """
        if task_type not in ARTIFACT_SCHEMA_MAP:
            raise ValueError(f"Unknown task type: {task_type}")

        schema_info = ARTIFACT_SCHEMA_MAP[task_type]
        artifact_type = schema_info["artifact_type"]
        schema_class = schema_info["schema"]
        result_key = schema_info["result_key"]

        # Extract provenance metadata from ML result
        config_hash = ml_result.get("config_hash")
        input_hash = ml_result.get("input_hash")
        run_id = ml_result.get("run_id")
        producer = ml_result.get("producer")
        producer_version = ml_result.get("producer_version")
        model_profile = ml_result.get("model_profile")

        if not all([config_hash, input_hash, run_id, producer, producer_version]):
            raise ValueError(
                f"ML result missing required provenance fields for task {task_id}"
            )

        # Extract items from result
        items = ml_result.get(result_key, [])

        # Special handling for metadata_extraction: wrap single dict as list
        if task_type == "metadata_extraction" and isinstance(items, dict):
            items = [items]

        if not isinstance(items, list):
            raise ValueError(
                f"Expected {result_key} to be a list in ML result for task {task_id}"
            )

        logger.info(
            f"Transforming {len(items)} items from ML result for task {task_id} "
            f"({task_type})"
        )

        envelopes = []
        for idx, item in enumerate(items):
            try:
                # Validate item against schema
                validated_item = schema_class(**item)

                # Create ArtifactEnvelope
                envelope = ArtifactTransformer._create_envelope(
                    task_id=task_id,
                    video_id=video_id,
                    artifact_type=artifact_type,
                    item=validated_item,
                    config_hash=config_hash,
                    input_hash=input_hash,
                    run_id=run_id,
                    producer=producer,
                    producer_version=producer_version,
                    model_profile=model_profile,
                )
                envelopes.append(envelope)

            except ValidationError as e:
                logger.error(
                    f"Schema validation failed for item {idx} in task {task_id}: {e}"
                )
                raise ValueError(
                    f"Invalid artifact payload at index {idx}: {e.error_count()} "
                    f"validation error(s)"
                ) from e

        logger.info(
            f"Successfully transformed {len(envelopes)} artifacts for task {task_id}"
        )
        return envelopes

    @staticmethod
    def _create_envelope(
        task_id: str,
        video_id: str,
        artifact_type: str,
        item: Any,
        config_hash: str,
        input_hash: str,
        run_id: str,
        producer: str,
        producer_version: str,
        model_profile: str,
    ) -> dict:
        """Create an ArtifactEnvelope from a validated item.

        Args:
            task_id: Task identifier
            video_id: Video identifier (asset_id)
            artifact_type: Type of artifact
            item: Validated schema object
            config_hash: Configuration hash
            input_hash: Input hash
            run_id: Run identifier
            producer: Producer name
            producer_version: Producer version
            model_profile: Model profile used

        Returns:
            ArtifactEnvelope dict ready for persistence
        """
        # Extract timing information based on artifact type
        span_start_ms, span_end_ms = ArtifactTransformer._extract_timing(
            artifact_type, item
        )

        # Convert validated item to dict for payload_json
        payload_json = json.loads(item.model_dump_json())

        return {
            "asset_id": video_id,
            "artifact_type": artifact_type,
            "schema_version": 1,
            "span_start_ms": span_start_ms,
            "span_end_ms": span_end_ms,
            "payload_json": payload_json,
            "config_hash": config_hash,
            "input_hash": input_hash,
            "producer": producer,
            "producer_version": producer_version,
            "model_profile": model_profile,
            "run_id": run_id,
        }

    @staticmethod
    def _extract_timing(artifact_type: str, item: Any) -> tuple[int, int]:
        """Extract timing information from artifact item.

        Args:
            artifact_type: Type of artifact
            item: Validated schema object

        Returns:
            Tuple of (span_start_ms, span_end_ms)
        """
        if artifact_type == "object.detection":
            # Object detection: use frame_number as both start and end
            # (single frame detection)
            frame_ms = item.frame_number * 33  # Approximate 30fps
            return frame_ms, frame_ms + 33

        elif artifact_type == "face.detection":
            # Face detection: use frame_number as both start and end
            frame_ms = item.frame_number * 33
            return frame_ms, frame_ms + 33

        elif artifact_type == "transcription":
            # Transcription: use start_ms and end_ms directly
            return item.start_ms, item.end_ms

        elif artifact_type == "ocr":
            # OCR: use frame_number as both start and end
            frame_ms = item.frame_number * 33
            return frame_ms, frame_ms + 33

        elif artifact_type == "place.classification":
            # Place classification: use frame_number as both start and end
            frame_ms = item.frame_number * 33
            return frame_ms, frame_ms + 33

        elif artifact_type == "scene.detection":
            # Scene detection: use start_ms and end_ms directly
            return item.start_ms, item.end_ms

        elif artifact_type == "video.metadata":
            # Metadata: spans entire video (0 to duration)
            # Duration is in seconds, convert to milliseconds
            duration_ms = (
                int(item.duration_seconds * 1000) if item.duration_seconds else 0
            )
            return 0, duration_ms

        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
