"""Artifact payload schema models for validation."""

from src.domain.schemas.face_detection_v1 import FaceDetectionV1
from src.domain.schemas.metadata_v1 import MetadataV1
from src.domain.schemas.object_detection_v1 import ObjectDetectionV1
from src.domain.schemas.ocr_detection_v1 import OCRDetectionV1
from src.domain.schemas.ocr_text_v1 import OcrTextV1
from src.domain.schemas.place_classification_v1 import PlaceClassificationV1
from src.domain.schemas.scene_v1 import SceneV1
from src.domain.schemas.transcript_segment_v1 import TranscriptSegmentV1

__all__ = [
    "ObjectDetectionV1",
    "FaceDetectionV1",
    "TranscriptSegmentV1",
    "OCRDetectionV1",
    "OcrTextV1",
    "PlaceClassificationV1",
    "SceneV1",
    "MetadataV1",
]
