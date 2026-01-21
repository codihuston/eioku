"""Artifact payload schemas for validation."""

from .face_detection_v1 import FaceDetectionV1
from .object_detection_v1 import ObjectDetectionV1
from .ocr_text_v1 import OcrTextV1
from .place_classification_v1 import PlaceClassificationV1
from .scene_v1 import SceneV1
from .transcript_segment_v1 import TranscriptSegmentV1

__all__ = [
    "TranscriptSegmentV1",
    "SceneV1",
    "ObjectDetectionV1",
    "FaceDetectionV1",
    "PlaceClassificationV1",
    "OcrTextV1",
]
