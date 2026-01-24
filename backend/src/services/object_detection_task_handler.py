"""Object detection task handler for video processing orchestration."""

import hashlib
import json
import logging
import uuid
from datetime import datetime

from ..domain.artifacts import ArtifactEnvelope
from ..domain.models import Task, Video
from ..domain.schema_registry import SchemaRegistry
from ..domain.schemas.object_detection_v1 import BoundingBox, ObjectDetectionV1
from ..repositories.interfaces import ArtifactRepository

logger = logging.getLogger(__name__)


class ObjectDetectionTaskHandler:
    """Handles object detection tasks in the orchestration system."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        schema_registry: SchemaRegistry,
        model_name: str = "yolov8n.pt",
        sample_rate: int = 30,
    ):
        self.artifact_repository = artifact_repository
        self.schema_registry = schema_registry
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.model = None  # Lazy load

    def _load_model(self):
        """Lazy load the YOLO model."""
        if self.model is None:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO object detection model: {self.model_name}")
            self.model = YOLO(self.model_name)

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

    def process_object_detection_task(
        self,
        task: Task,
        video: Video,
        run_id: str | None = None,
        model_profile: str | None = None,
    ) -> bool:
        """Process an object detection task for a video.

        Args:
            task: The object detection task to process
            video: The video to analyze
            run_id: Optional run ID for tracking (generated if not provided)
            model_profile: Optional model profile (fast, balanced, high_quality).
                          If not provided, determined from model name.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting object detection for video {video.video_id}")

            # Generate run_id if not provided
            if run_id is None:
                run_id = str(uuid.uuid4())
                logger.info(f"Generated run_id: {run_id}")

            # Lazy load model
            self._load_model()

            # Detect objects in video using YOLO directly
            import cv2

            cap = cv2.VideoCapture(video.file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Group detections by label for aggregation
            detections_by_label = {}

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames based on sample_rate
                if frame_idx % self.sample_rate == 0:
                    # Run YOLO detection
                    results = self.model(frame, verbose=False)

                    # Process detections
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Get detection info
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            label = self.model.names[class_id]

                            # Store detection
                            if label not in detections_by_label:
                                detections_by_label[label] = []

                            timestamp_sec = frame_idx / fps
                            detections_by_label[label].append(
                                {
                                    "frame": frame_idx,
                                    "timestamp": timestamp_sec,
                                    "bbox": [
                                        float(x1),
                                        float(y1),
                                        float(x2),
                                        float(y2),
                                    ],
                                    "confidence": confidence,
                                }
                            )

                frame_idx += 1

            cap.release()

            logger.info(f"Detected {len(detections_by_label)} unique object types")

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

            # Convert detections to individual artifact envelopes
            # Create one artifact per detection (frame-level granularity)
            artifacts = []
            for label, bbox_list in detections_by_label.items():
                # Each label has multiple bounding boxes (one per frame)
                for bbox_data in bbox_list:
                    frame_number = bbox_data["frame"]
                    timestamp_sec = bbox_data["timestamp"]
                    bbox_coords = bbox_data["bbox"]  # [x1, y1, x2, y2]
                    confidence = bbox_data["confidence"]

                    # Convert YOLO bbox format [x1, y1, x2, y2] to [x, y, width, height]
                    x1, y1, x2, y2 = bbox_coords
                    bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

                    # Create payload using Pydantic schema
                    payload = ObjectDetectionV1(
                        label=label,
                        confidence=confidence,
                        bounding_box=bbox,
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
                        artifact_type="object.detection",
                        schema_version=1,
                        span_start_ms=span_start_ms,
                        span_end_ms=span_end_ms,
                        payload_json=payload.model_dump_json(),
                        producer="yolo",
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
                f"Object detection complete for video {video.video_id}. "
                f"Saved {saved_count} object detection artifacts"
            )
            return True

        except Exception as e:
            logger.error(f"Object detection failed for video {video.video_id}: {e}")
            return False

    def get_detected_objects(self, video_id: str) -> list[ArtifactEnvelope]:
        """Get all detected objects for a video.

        Args:
            video_id: Video ID

        Returns:
            List of object detection artifacts
        """
        return self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="object.detection"
        )

    def get_objects_by_label(self, video_id: str, label: str) -> list[ArtifactEnvelope]:
        """Get detected objects filtered by label.

        Args:
            video_id: Video ID
            label: Object label to filter by

        Returns:
            List of object detection artifacts with the specified label
        """
        artifacts = self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="object.detection"
        )

        # Filter by label
        matching_artifacts = []
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            if payload.get("label") == label:
                matching_artifacts.append(artifact)

        return matching_artifacts
