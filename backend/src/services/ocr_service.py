"""OCR text detection service using EasyOCR."""

import logging
from pathlib import Path

import cv2
import easyocr

logger = logging.getLogger(__name__)


class OcrService:
    """Service for detecting and extracting text from video frames using EasyOCR."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        """Initialize the OCR service.

        Args:
            languages: List of language codes to detect (default: ['en'])
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.reader = None
        self._load_reader()

    def _load_reader(self):
        """Load the EasyOCR reader."""
        try:
            logger.info(
                f"Loading EasyOCR reader for languages: {self.languages}, "
                f"GPU: {self.gpu}"
            )
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR reader loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR reader: {e}")
            raise

    def detect_text_in_frame(self, frame) -> list[dict]:
        """Detect text in a single frame.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            List of dicts with 'text', 'confidence', 'bounding_box', and 'language'
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run OCR
            results = self.reader.readtext(rgb_frame)

            # Parse results
            detections = []
            for bbox, text, confidence in results:
                # Skip empty text
                if not text.strip():
                    continue

                # Convert bbox to polygon points
                # bbox is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                polygon_points = [{"x": float(p[0]), "y": float(p[1])} for p in bbox]

                detections.append(
                    {
                        "text": text,
                        "confidence": float(confidence),
                        "bounding_box": polygon_points,
                        "language": self.languages[
                            0
                        ],  # EasyOCR doesn't return language per detection
                    }
                )

            return detections

        except Exception as e:
            logger.error(f"Failed to detect text in frame: {e}")
            return []

    def detect_text_in_video(
        self, video_path: str, sample_rate: int = 30
    ) -> list[dict]:
        """Detect text in a video by sampling frames.

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame

        Returns:
            List of dicts with frame_number, timestamp, and detections
        """
        results = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return results

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames at specified rate
                if frame_idx % sample_rate == 0:
                    timestamp_sec = frame_idx / fps
                    detections = self.detect_text_in_frame(frame)

                    if detections:
                        results.append(
                            {
                                "frame_number": frame_idx,
                                "timestamp": timestamp_sec,
                                "detections": detections,
                            }
                        )

                frame_idx += 1

            cap.release()
            logger.info(
                f"Processed {len(results)} frames from video {Path(video_path).name}"
            )

        except Exception as e:
            logger.error(f"Failed to process video {video_path}: {e}")

        return results
