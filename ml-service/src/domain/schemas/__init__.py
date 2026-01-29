"""Artifact payload schema models for validation."""

# Import schemas from backend to avoid duplication
# In production, these could be in a shared package
from pydantic import BaseModel, Field


class Word(BaseModel):
    """Word-level transcription detail."""

    word: str = Field(..., description="The word text")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Word confidence score"
    )


class TranscriptSegmentV1(BaseModel):
    """Payload schema for transcription artifacts."""

    text: str = Field(..., description="Transcribed text")
    start_ms: int = Field(..., ge=0, description="Start time in milliseconds")
    end_ms: int = Field(..., ge=0, description="End time in milliseconds")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Segment confidence score"
    )
    words: list[Word] | None = Field(
        default=None, description="Word-level transcription details"
    )


class SceneV1(BaseModel):
    """Payload schema for scene detection artifacts."""

    scene_index: int = Field(..., ge=0, description="Scene index")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Scene confidence score"
    )


class ObjectDetectionV1(BaseModel):
    """Payload schema for object detection artifacts."""

    label: str = Field(..., description="Object label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: dict | None = Field(default=None, description="Bounding box coordinates")


class FaceDetectionV1(BaseModel):
    """Payload schema for face detection artifacts."""

    cluster_id: str | None = Field(default=None, description="Face cluster ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    embedding: list[float] | None = Field(default=None, description="Face embedding")


class PlaceClassificationV1(BaseModel):
    """Payload schema for place classification artifacts."""

    label: str = Field(..., description="Place label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )


class OCRDetectionV1(BaseModel):
    """Payload schema for OCR detection artifacts."""

    text: str = Field(..., description="Detected text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: dict | None = Field(default=None, description="Bounding box coordinates")


class OcrTextV1(BaseModel):
    """Payload schema for OCR text artifacts."""

    text: str = Field(..., description="Extracted text")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Text confidence score"
    )


__all__ = [
    "ObjectDetectionV1",
    "FaceDetectionV1",
    "TranscriptSegmentV1",
    "OCRDetectionV1",
    "OcrTextV1",
    "PlaceClassificationV1",
    "SceneV1",
]
