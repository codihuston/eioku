"""Place detection task handler for video processing orchestration."""

import hashlib
import json
import logging
import uuid
from datetime import datetime

from ..domain.artifacts import ArtifactEnvelope
from ..domain.models import Task, Video
from ..domain.schema_registry import SchemaRegistry
from ..domain.schemas.place_classification_v1 import (
    AlternativeLabel,
    PlaceClassificationV1,
)
from ..repositories.interfaces import ArtifactRepository
from .place_detection_service import PlaceDetectionService

logger = logging.getLogger(__name__)


class PlaceDetectionTaskHandler:
    """Handles place classification tasks in the orchestration system."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        schema_registry: SchemaRegistry,
        detection_service: PlaceDetectionService | None = None,
        model_name: str = "resnet18_places365",
        sample_rate: int = 30,
        top_k: int = 5,
    ):
        """Initialize the place detection task handler.

        Args:
            artifact_repository: Repository for storing artifacts
            schema_registry: Schema registry for validation
            detection_service: Optional place detection service instance
            model_name: Name of the model for provenance tracking
            sample_rate: Process every Nth frame
            top_k: Number of top predictions to store per frame
        """
        self.artifact_repository = artifact_repository
        self.schema_registry = schema_registry
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.detection_service = detection_service or PlaceDetectionService()

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
        # ResNet18 is relatively fast
        # Could add ResNet50 or ResNet152 for higher quality
        if "resnet152" in model_name.lower():
            return "high_quality"
        elif "resnet50" in model_name.lower():
            return "balanced"
        else:
            return "fast"

    def process_place_detection_task(
        self,
        task: Task,
        video: Video,
        run_id: str | None = None,
        model_profile: str | None = None,
    ) -> bool:
        """Process a place detection task for a video.

        Args:
            task: The place detection task to process
            video: The video to analyze
            run_id: Optional run ID for tracking (generated if not provided)
            model_profile: Optional model profile (fast, balanced, high_quality).
                          If not provided, determined from model name.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting place detection for video {video.video_id}")

            # Generate run_id if not provided
            if run_id is None:
                run_id = str(uuid.uuid4())
                logger.info(f"Generated run_id: {run_id}")

            # Detect places in video using configured sample rate
            frame_results = self.detection_service.detect_places_in_video(
                video_path=video.file_path,
                sample_rate=self.sample_rate,
                top_k=self.top_k,
            )

            logger.info(f"Classified {len(frame_results)} frames")

            # Compute provenance hashes
            config = {
                "model_name": self.model_name,
                "sample_rate": self.sample_rate,
                "top_k": self.top_k,
            }
            config_hash = self._compute_config_hash(config)
            input_hash = self._compute_input_hash(video.file_path)

            # Determine model profile - use provided or infer from model name
            if model_profile is None:
                model_profile = self._determine_model_profile(self.model_name)

            # Create one artifact per frame classification
            artifacts = []
            for frame_result in frame_results:
                frame_number = frame_result["frame_number"]
                timestamp_sec = frame_result["timestamp"]
                classifications = frame_result["classifications"]

                if not classifications:
                    continue

                # Primary classification is the first (highest confidence)
                primary = classifications[0]

                # Alternative labels are the rest
                alternative_labels = [
                    AlternativeLabel(label=c["label"], confidence=c["confidence"])
                    for c in classifications[1:]
                ]

                # Create payload using Pydantic schema
                payload = PlaceClassificationV1(
                    label=primary["label"],
                    confidence=primary["confidence"],
                    alternative_labels=alternative_labels,
                    frame_number=frame_number,
                )

                # Calculate time span for this classification
                # Use a small window around the classification timestamp
                span_start_ms = int(timestamp_sec * 1000)
                span_end_ms = span_start_ms + 1  # 1ms duration for frame-level

                # Create artifact envelope
                artifact = ArtifactEnvelope(
                    artifact_id=str(uuid.uuid4()),
                    asset_id=video.video_id,
                    artifact_type="place.classification",
                    schema_version=1,
                    span_start_ms=span_start_ms,
                    span_end_ms=span_end_ms,
                    payload_json=payload.model_dump_json(),
                    producer="resnet_places365",
                    producer_version=self.model_name,
                    model_profile=model_profile,
                    config_hash=config_hash,
                    input_hash=input_hash,
                    run_id=run_id,
                    created_at=datetime.utcnow(),
                )

                artifacts.append(artifact)

            # Batch insert all artifacts
            self.artifact_repository.batch_create(artifacts)
            saved_count = len(artifacts)

            logger.info(
                f"Place detection complete for video {video.video_id}. "
                f"Saved {saved_count} place classification artifacts"
            )
            return True

        except Exception as e:
            logger.error(f"Place detection failed for video {video.video_id}: {e}")
            return False

    def get_detected_places(self, video_id: str) -> list[ArtifactEnvelope]:
        """Get all detected places for a video.

        Args:
            video_id: Video ID

        Returns:
            List of place classification artifacts
        """
        return self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="place.classification"
        )

    def get_places_by_label(self, video_id: str, label: str) -> list[ArtifactEnvelope]:
        """Get detected places filtered by label.

        Args:
            video_id: Video ID
            label: Place label to filter by

        Returns:
            List of place classification artifacts with the specified label
        """
        artifacts = self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="place.classification"
        )

        # Filter by label
        matching_artifacts = []
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            if payload.get("label") == label:
                matching_artifacts.append(artifact)

        return matching_artifacts
