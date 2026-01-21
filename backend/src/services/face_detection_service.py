"""Face detection service using YOLOv8-face."""

import uuid
from pathlib import Path

import av

from ..domain.models import Face
from ..utils.print_logger import get_logger

logger = get_logger(__name__)


class FaceDetectionError(Exception):
    """Exception raised for face detection errors."""

    pass


class FaceDetectionService:
    """Service for detecting faces in video frames using YOLOv8-face."""

    def __init__(self, model_name: str = "yolov8n-face.pt"):
        """Initialize face detection service.

        Args:
            model_name: YOLOv8-face model to use
                (yolov8n-face.pt, yolov8s-face.pt, etc.)
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize YOLO face detection model."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO face detection model: {self.model_name}")
            self.model = YOLO(self.model_name)
            logger.info("YOLO face detection model loaded successfully")
        except ImportError as e:
            raise FaceDetectionError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from e
        except Exception as e:
            raise FaceDetectionError(
                f"Failed to load YOLO face detection model: {e}"
            ) from e

    def detect_faces_in_video(
        self, video_path: str, video_id: str, sample_rate: int = 30
    ) -> list[Face]:
        """Detect faces in video frames.

        Args:
            video_path: Path to video file
            video_id: Video ID for associating detections
            sample_rate: Process every Nth frame
                (default: 30 = 1 frame per second at 30fps)

        Returns:
            List of Face domain models with detections
        """
        if not Path(video_path).exists():
            raise FaceDetectionError(f"Video file not found: {video_path}")

        logger.info(f"Starting face detection for video: {video_path}")
        logger.info(f"Sample rate: every {sample_rate} frames")

        try:
            container = av.open(video_path)
        except av.AVError as e:
            raise FaceDetectionError(f"Failed to open video: {e}") from e

        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)

        logger.info(
            f"Video info: {video_stream.codec_context.name} codec, "
            f"{fps:.2f} fps, {video_stream.frames} frames"
        )

        frame_idx = 0
        processed_frames = 0

        # Dictionary to aggregate detections by person_id (cluster)
        # person_id -> {timestamps: [], bounding_boxes: [], confidences: []}
        detections_by_person = {}

        try:
            for frame in container.decode(video=0):
                # Sample frames based on sample_rate
                if frame_idx % sample_rate == 0:
                    # Convert PyAV frame to numpy array (RGB format)
                    img = frame.to_ndarray(format="rgb24")

                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    self._process_frame(img, timestamp, detections_by_person, frame_idx)
                    processed_frames += 1

                    if processed_frames % 100 == 0:
                        logger.debug(
                            f"Processed {processed_frames} frames, "
                            f"found {len(detections_by_person)} unique faces"
                        )

                frame_idx += 1

        finally:
            container.close()

        logger.info(
            f"Face detection complete. Processed {processed_frames} frames, "
            f"found {len(detections_by_person)} unique face clusters"
        )

        # Convert aggregated detections to Face domain models
        faces = []
        for person_id, data in detections_by_person.items():
            # Calculate average confidence
            avg_confidence = (
                sum(data["confidences"]) / len(data["confidences"])
                if data["confidences"]
                else 0.0
            )

            face = Face(
                face_id=str(uuid.uuid4()),
                video_id=video_id,
                person_id=person_id,
                timestamps=data["timestamps"],
                bounding_boxes=data["bounding_boxes"],
                confidence=avg_confidence,
            )
            faces.append(face)

        return faces

    def _process_frame(
        self,
        frame,
        timestamp: float,
        detections_by_person: dict,
        frame_idx: int,
    ):
        """Process a single frame and aggregate detections.

        Args:
            frame: Video frame (numpy array)
            timestamp: Timestamp in seconds
            detections_by_person: Dictionary to aggregate detections
            frame_idx: Frame index for logging
        """
        try:
            results = self.model(frame, verbose=False)

            for r in results:
                for idx, box in enumerate(r.boxes):
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # For now, use detection index as person_id
                    # In production, this would be replaced with face clustering
                    person_id = f"face_{idx}"

                    # Initialize person entry if first occurrence
                    if person_id not in detections_by_person:
                        detections_by_person[person_id] = {
                            "timestamps": [],
                            "bounding_boxes": [],
                            "confidences": [],
                        }

                    # Add detection
                    detections_by_person[person_id]["timestamps"].append(timestamp)
                    detections_by_person[person_id]["bounding_boxes"].append(
                        {
                            "frame": frame_idx,
                            "timestamp": timestamp,
                            "bbox": xyxy,  # [x1, y1, x2, y2]
                            "confidence": conf,
                        }
                    )
                    detections_by_person[person_id]["confidences"].append(conf)

        except Exception as e:
            logger.warning(f"Error processing frame {frame_idx}: {e}")
