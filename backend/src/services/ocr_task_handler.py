"""OCR text detection task handler for video processing orchestration."""

import hashlib
import json
import logging
import uuid
from datetime import datetime

from ..domain.artifacts import ArtifactEnvelope
from ..domain.models import Task, Video
from ..domain.schema_registry import SchemaRegistry
from ..domain.schemas.ocr_text_v1 import OcrTextV1, PolygonPoint
from ..repositories.interfaces import ArtifactRepository
from .ocr_service import OcrService

logger = logging.getLogger(__name__)


class OcrTaskHandler:
    """Handles OCR text detection tasks in the orchestration system."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        schema_registry: SchemaRegistry,
        ocr_service: OcrService | None = None,
        languages: list[str] | None = None,
        sample_rate: int = 30,
        gpu: bool = False,
    ):
        """Initialize the OCR task handler.

        Args:
            artifact_repository: Repository for storing artifacts
            schema_registry: Schema registry for validation
            ocr_service: Optional OCR service instance
            languages: List of language codes to detect (default: ['en'])
            sample_rate: Process every Nth frame
            gpu: Whether to use GPU acceleration
        """
        self.artifact_repository = artifact_repository
        self.schema_registry = schema_registry
        self.languages = languages or ["en"]
        self.sample_rate = sample_rate
        self.gpu = gpu
        self.ocr_service = ocr_service or OcrService(
            languages=self.languages, gpu=self.gpu
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

    def _determine_model_profile(self, gpu: bool) -> str:
        """Determine model profile based on GPU usage."""
        # GPU is faster, CPU is slower
        return "fast" if gpu else "balanced"

    def process_ocr_task(
        self,
        task: Task,
        video: Video,
        run_id: str | None = None,
        model_profile: str | None = None,
    ) -> bool:
        """Process an OCR text detection task for a video.

        Args:
            task: The OCR task to process
            video: The video to analyze
            run_id: Optional run ID for tracking (generated if not provided)
            model_profile: Optional model profile (fast, balanced, high_quality).
                          If not provided, determined from GPU usage.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting OCR text detection for video {video.video_id}")

            # Generate run_id if not provided
            if run_id is None:
                run_id = str(uuid.uuid4())
                logger.info(f"Generated run_id: {run_id}")

            # Detect text in video using configured sample rate
            frame_results = self.ocr_service.detect_text_in_video(
                video_path=video.file_path, sample_rate=self.sample_rate
            )

            logger.info(f"Detected text in {len(frame_results)} frames")

            # Compute provenance hashes
            config = {
                "languages": self.languages,
                "sample_rate": self.sample_rate,
                "gpu": self.gpu,
            }
            config_hash = self._compute_config_hash(config)
            input_hash = self._compute_input_hash(video.file_path)

            # Determine model profile - use provided or infer from GPU usage
            if model_profile is None:
                model_profile = self._determine_model_profile(self.gpu)

            # Create one artifact per text detection
            saved_count = 0
            for frame_result in frame_results:
                frame_number = frame_result["frame_number"]
                timestamp_sec = frame_result["timestamp"]
                detections = frame_result["detections"]

                for detection in detections:
                    # Convert bounding box to PolygonPoint objects
                    bounding_box = [
                        PolygonPoint(x=point["x"], y=point["y"])
                        for point in detection["bounding_box"]
                    ]

                    # Create payload using Pydantic schema
                    payload = OcrTextV1(
                        text=detection["text"],
                        confidence=detection["confidence"],
                        bounding_box=bounding_box,
                        language=detection["language"],
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
                        artifact_type="ocr.text",
                        schema_version=1,
                        span_start_ms=span_start_ms,
                        span_end_ms=span_end_ms,
                        payload_json=payload.model_dump_json(),
                        producer="easyocr",
                        producer_version=f"easyocr_{'+'.join(self.languages)}",
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
                f"OCR text detection complete for video {video.video_id}. "
                f"Saved {saved_count} OCR text artifacts"
            )
            return True

        except Exception as e:
            logger.error(f"OCR text detection failed for video {video.video_id}: {e}")
            return False

    def get_detected_text(self, video_id: str) -> list[ArtifactEnvelope]:
        """Get all detected text for a video.

        Args:
            video_id: Video ID

        Returns:
            List of OCR text artifacts
        """
        return self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="ocr.text"
        )

    def get_text_by_content(
        self, video_id: str, search_text: str
    ) -> list[ArtifactEnvelope]:
        """Get detected text filtered by content.

        Args:
            video_id: Video ID
            search_text: Text to search for (case-insensitive substring match)

        Returns:
            List of OCR text artifacts containing the search text
        """
        artifacts = self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="ocr.text"
        )

        # Filter by text content
        matching_artifacts = []
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            if search_text.lower() in payload.get("text", "").lower():
                matching_artifacts.append(artifact)

        return matching_artifacts
