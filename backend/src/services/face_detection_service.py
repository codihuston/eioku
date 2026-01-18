"""Face detection service using YOLOv8 face model."""

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
    """Service for detecting faces in video frames using YOLOv8 face model."""

    def __init__(self, model_name: str = "yolov8n-face.pt"):
        """Initialize face detection service.

        Args:
            model_name: YOLOv8 face model to use
                (yolov8n-face.pt, yolov8s-face.pt, etc.)
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize YOLO face model."""
        try:
            from ultralytics import YOLO

            # Use models directory if it exists, otherwise use current directory
            model_path = Path("models") / self.model_name
            if not model_path.exists():
                model_path = Path(self.model_name)

            logger.info(f"Loading YOLO face model from: {model_path}")
            self.model = YOLO(str(model_path))
            logger.info("YOLO face model loaded successfully")
        except ImportError as e:
            raise FaceDetectionError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from e
        except Exception as e:
            raise FaceDetectionError(f"Failed to load YOLO face model: {e}") from e

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
            (single Face object with all detections)
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

        # Aggregate all face detections into single list
        all_timestamps = []
        all_bounding_boxes = []
        confidence_sum = 0.0
        detection_count = 0

        try:
            for frame in container.decode(video=0):
                # Sample frames based on sample_rate
                if frame_idx % sample_rate == 0:
                    # Convert PyAV frame to numpy array (RGB format)
                    img = frame.to_ndarray(format="rgb24")

                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    self._process_frame(
                        img,
                        timestamp,
                        all_timestamps,
                        all_bounding_boxes,
                        frame_idx,
                    )

                    # Update confidence tracking
                    if all_bounding_boxes:
                        # Get confidence from last added bounding box
                        last_conf = all_bounding_boxes[-1]["confidence"]
                        confidence_sum += last_conf
                        detection_count += 1

                    processed_frames += 1

                    if processed_frames % 100 == 0:
                        logger.debug(
                            f"Processed {processed_frames} frames, "
                            f"found {len(all_timestamps)} face detections"
                        )

                frame_idx += 1

        finally:
            container.close()

        logger.info(
            f"Face detection complete. Processed {processed_frames} frames, "
            f"found {len(all_timestamps)} face detections"
        )

        # Calculate average confidence
        avg_confidence = (
            confidence_sum / detection_count if detection_count > 0 else 0.0
        )

        # Create single Face object with all detections
        # person_id is None (no clustering yet - Phase 2)
        faces = []
        if all_timestamps:
            face = Face(
                face_id=str(uuid.uuid4()),
                video_id=video_id,
                person_id=None,  # No clustering yet (Phase 2)
                timestamps=all_timestamps,
                bounding_boxes=all_bounding_boxes,
                confidence=avg_confidence,
            )
            faces.append(face)

        return faces

    def _process_frame(
        self,
        frame,
        timestamp: float,
        all_timestamps: list,
        all_bounding_boxes: list,
        frame_idx: int,
    ):
        """Process a single frame and aggregate face detections.

        Args:
            frame: Video frame (numpy array)
            timestamp: Timestamp in seconds
            all_timestamps: List to append timestamps
            all_bounding_boxes: List to append bounding boxes
            frame_idx: Frame index for logging
        """
        try:
            results = self.model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    # YOLOv8 face model typically has single class (face)
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # Add detection to aggregated lists
                    all_timestamps.append(timestamp)
                    all_bounding_boxes.append(
                        {
                            "frame": frame_idx,
                            "timestamp": timestamp,
                            "bbox": xyxy,  # [x1, y1, x2, y2]
                            "confidence": conf,
                        }
                    )

        except Exception as e:
            logger.warning(f"Error processing frame {frame_idx}: {e}")
