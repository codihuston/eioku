"""Face detection task handler for video processing orchestration."""

import hashlib
import json
import logging
import uuid
from datetime import datetime

from ..domain.artifacts import ArtifactEnvelope
from ..domain.models import Task, Video
from ..domain.schema_registry import SchemaRegistry
from ..domain.schemas.face_detection_v1 import BoundingBox, FaceDetectionV1
from ..repositories.interfaces import ArtifactRepository
from .face_detection_service import FaceDetectionService

logger = logging.getLogger(__name__)


class FaceDetectionTaskHandler:
    """Handles face detection tasks in the orchestration system."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        schema_registry: SchemaRegistry,
        detection_service: FaceDetectionService | None = None,
        model_name: str = "yolov8n-face.pt",
        sample_rate: int = 30,
    ):
        self.artifact_repository = artifact_repository
        self.schema_registry = schema_registry
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.detection_service = detection_service or FaceDetectionService(
            model_name=model_name
        )

    def _compute_config_hash(self, config: dict) -> str:
        """Compute hash of configuration for provenance tracking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _compute_input_hash(self, video_path: str) -> str:
        """Compute hash of input video file for provenance tracking."""
        # For now, use video path as input identifier
        # In production, could use file hash or video_id
        return hashlib.sha256(video_path.encode()).hexdigest()[:16]

    def _determine_model_profile(self, model_name: str) -> str:
        """Determine model profile based on model name."""
        if "yolov8x" in model_name or "yolov8l" in model_name:
            return "high_quality"
        elif "yolov8m" in model_name:
            return "balanced"
        else:
            return "fast"

    def process_face_detection_task(
        self,
        task: Task,
        video: Video,
        run_id: str | None = None,
        model_profile: str | None = None,
    ) -> bool:
        """Process a face detection task for a video.

        Args:
            task: The face detection task to process
            video: The video to analyze
            run_id: Optional run ID for tracking (generated if not provided)
            model_profile: Optional model profile (fast, balanced, high_quality).
                          If not provided, determined from model name.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting face detection for video {video.video_id}")

            # Generate run_id if not provided
            if run_id is None:
                run_id = str(uuid.uuid4())
                logger.info(f"Generated run_id: {run_id}")

            # Detect faces in video using configured sample rate
            # This returns the old Face domain models with aggregated detections
            legacy_faces = self.detection_service.detect_faces_in_video(
                video_path=video.file_path,
                video_id=video.video_id,
                sample_rate=self.sample_rate,
            )

            logger.info(f"Detected {len(legacy_faces)} unique face clusters")

            # Compute provenance hashes
            config = {
                "model_name": self.model_name,
                "sample_rate": self.sample_rate,
            }
            config_hash = self._compute_config_hash(config)
            input_hash = self._compute_input_hash(video.file_path)

            # Determine model profile - use provided or infer from model name
            if model_profile is None:
                model_profile = self._determine_model_profile(self.model_name)

            # Convert legacy aggregated faces to individual artifact envelopes
            # Create one artifact per detection (frame-level granularity)
            saved_count = 0
            for legacy_face in legacy_faces:
                # Each legacy face has multiple bounding boxes (one per frame)
                for bbox_data in legacy_face.bounding_boxes:
                    frame_number = bbox_data["frame"]
                    timestamp_sec = bbox_data["timestamp"]
                    bbox_coords = bbox_data["bbox"]  # [x1, y1, x2, y2]
                    confidence = bbox_data["confidence"]

                    # Convert YOLO bbox format [x1, y1, x2, y2] to [x, y, width, height]
                    x1, y1, x2, y2 = bbox_coords
                    bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

                    # Create payload using Pydantic schema
                    payload = FaceDetectionV1(
                        confidence=confidence,
                        bounding_box=bbox,
                        cluster_id=legacy_face.person_id,
                        frame_number=frame_number,
                    )

                    # Calculate time span for this detection
                    # Use a small window around the detection timestamp
                    span_start_ms = int(timestamp_sec * 1000)
                    span_end_ms = span_start_ms + 1  # 1ms duration for frame-level

                    # Create artifact envelope
                    artifact = ArtifactEnvelope(
                        artifact_id=str(uuid.uuid4()),
                        asset_id=video.video_id,
                        artifact_type="face.detection",
                        schema_version=1,
                        span_start_ms=span_start_ms,
                        span_end_ms=span_end_ms,
                        payload_json=payload.model_dump_json(),
                        producer="yolo-face",
                        producer_version=self.model_name,
                        model_profile=model_profile,
                        config_hash=config_hash,
                        input_hash=input_hash,
                        run_id=run_id,
                        created_at=datetime.utcnow(),
                    )

                    # Save to artifact repository
                    self.artifact_repository.create(artifact)
                    saved_count += 1

            logger.info(
                f"Face detection complete for video {video.video_id}. "
                f"Saved {saved_count} face detection artifacts"
            )
            return True

        except Exception as e:
            logger.error(f"Face detection failed for video {video.video_id}: {e}")
            return False

    def get_detected_faces(self, video_id: str) -> list[ArtifactEnvelope]:
        """Get all detected faces for a video.

        Args:
            video_id: Video ID

        Returns:
            List of face detection artifacts
        """
        return self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="face.detection"
        )

    def get_faces_by_cluster(
        self, video_id: str, cluster_id: str
    ) -> list[ArtifactEnvelope]:
        """Get detected faces filtered by cluster ID.

        Args:
            video_id: Video ID
            cluster_id: Face cluster ID to filter by

        Returns:
            List of face detection artifacts with the specified cluster ID
        """
        artifacts = self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="face.detection"
        )

        # Filter by cluster_id
        matching_artifacts = []
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            if payload.get("cluster_id") == cluster_id:
                matching_artifacts.append(artifact)

        return matching_artifacts
