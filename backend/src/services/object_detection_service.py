"""Object detection service using YOLOv8."""

import uuid
from pathlib import Path

import av

from ..domain.models import Object
from ..utils.print_logger import get_logger

logger = get_logger(__name__)


class ObjectDetectionError(Exception):
    """Exception raised for object detection errors."""

    pass


class ObjectDetectionService:
    """Service for detecting objects in video frames using YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize object detection service.

        Args:
            model_name: YOLOv8 model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize YOLO model."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            logger.info("YOLO model loaded successfully")
        except ImportError as e:
            raise ObjectDetectionError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from e
        except Exception as e:
            raise ObjectDetectionError(f"Failed to load YOLO model: {e}") from e

    def detect_objects_in_video(
        self, video_path: str, video_id: str, sample_rate: int = 30
    ) -> list[Object]:
        """Detect objects in video frames.

        Args:
            video_path: Path to video file
            video_id: Video ID for associating detections
            sample_rate: Process every Nth frame
                (default: 30 = 1 frame per second at 30fps)

        Returns:
            List of Object domain models with detections
        """
        if not Path(video_path).exists():
            raise ObjectDetectionError(f"Video file not found: {video_path}")

        logger.info(f"Starting object detection for video: {video_path}")
        logger.info(f"Sample rate: every {sample_rate} frames")

        try:
            container = av.open(video_path)
        except av.AVError as e:
            raise ObjectDetectionError(f"Failed to open video: {e}") from e

        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)

        logger.info(
            f"Video info: {video_stream.codec_context.name} codec, "
            f"{fps:.2f} fps, {video_stream.frames} frames"
        )

        frame_idx = 0
        processed_frames = 0

        # Dictionary to aggregate detections by label
        # label -> {timestamps: [], bounding_boxes: []}
        detections_by_label = {}

        try:
            for frame in container.decode(video=0):
                # Sample frames based on sample_rate
                if frame_idx % sample_rate == 0:
                    # Convert PyAV frame to numpy array (RGB format)
                    img = frame.to_ndarray(format="rgb24")

                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    self._process_frame(img, timestamp, detections_by_label, frame_idx)
                    processed_frames += 1

                    if processed_frames % 100 == 0:
                        logger.debug(
                            f"Processed {processed_frames} frames, "
                            f"found {len(detections_by_label)} unique labels"
                        )

                frame_idx += 1

        finally:
            container.close()

        logger.info(
            f"Object detection complete. Processed {processed_frames} frames, "
            f"found {len(detections_by_label)} unique object labels"
        )

        # Convert aggregated detections to Object domain models
        objects = []
        for label, data in detections_by_label.items():
            obj = Object(
                object_id=str(uuid.uuid4()),
                video_id=video_id,
                label=label,
                timestamps=data["timestamps"],
                bounding_boxes=data["bounding_boxes"],
            )
            objects.append(obj)

        return objects

    def _process_frame(
        self,
        frame,
        timestamp: float,
        detections_by_label: dict,
        frame_idx: int,
    ):
        """Process a single frame and aggregate detections.

        Args:
            frame: Video frame (numpy array)
            timestamp: Timestamp in seconds
            detections_by_label: Dictionary to aggregate detections
            frame_idx: Frame index for logging
        """
        try:
            results = self.model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # Initialize label entry if first occurrence
                    if label not in detections_by_label:
                        detections_by_label[label] = {
                            "timestamps": [],
                            "bounding_boxes": [],
                        }

                    # Add detection
                    detections_by_label[label]["timestamps"].append(timestamp)
                    detections_by_label[label]["bounding_boxes"].append(
                        {
                            "frame": frame_idx,
                            "timestamp": timestamp,
                            "bbox": xyxy,  # [x1, y1, x2, y2]
                            "confidence": conf,
                        }
                    )

        except Exception as e:
            logger.warning(f"Error processing frame {frame_idx}: {e}")
