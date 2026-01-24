"""Face detection service using YOLO."""

import cv2
from ultralytics import YOLO

from ..utils.print_logger import get_logger

logger = get_logger(__name__)


class FaceDetectionError(Exception):
    """Exception raised when face detection fails."""

    pass


class FaceDetectionService:
    """Service for detecting faces in videos using YOLO."""

    def __init__(self, model_name: str = "yolov8n-face.pt"):
        """Initialize face detection service.

        Args:
            model_name: Name of the YOLO model to use for face detection
        """
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Lazy load the YOLO model."""
        if self.model is None:
            logger.info(f"Loading YOLO face detection model: {self.model_name}")
            self.model = YOLO(self.model_name)

    def detect_faces_in_video(
        self, video_path: str, sample_rate: int = 30
    ) -> list[dict]:
        """Detect faces in a video file.

        Args:
            video_path: Path to the video file
            sample_rate: Sample every Nth frame

        Returns:
            List of frame-level face detections with format:
            [
                {
                    "frame_number": int,
                    "timestamp": float,
                    "detections": [
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float
                        },
                        ...
                    ]
                },
                ...
            ]

        Raises:
            FaceDetectionError: If face detection fails
        """
        self._load_model()

        logger.info(f"Detecting faces in video: {video_path}")

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_results = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames based on sample_rate
                if frame_idx % sample_rate == 0:
                    timestamp_sec = frame_idx / fps

                    # Run YOLO detection
                    results = self.model(frame, verbose=False)

                    # Process detections for this frame
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())

                            detections.append(
                                {
                                    "bbox": [
                                        float(x1),
                                        float(y1),
                                        float(x2),
                                        float(y2),
                                    ],
                                    "confidence": confidence,
                                }
                            )

                    # Only add frame if we detected faces
                    if detections:
                        frame_results.append(
                            {
                                "frame_number": frame_idx,
                                "timestamp": timestamp_sec,
                                "detections": detections,
                            }
                        )

                frame_idx += 1

            cap.release()

            logger.info(f"Detected faces in {len(frame_results)} frames")
            return frame_results

        except Exception as e:
            error_msg = f"Face detection failed for {video_path}: {str(e)}"
            logger.error(error_msg)
            raise FaceDetectionError(error_msg)
