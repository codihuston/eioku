"""Request models for ML Service inference endpoints."""

from pydantic import BaseModel, Field


class ObjectDetectionRequest(BaseModel):
    """Request model for object detection inference."""

    video_path: str = Field(..., description="Path to video file")
    model_name: str = Field(default="yolov8n.pt", description="Model file name")
    frame_interval: int = Field(default=30, description="Process every Nth frame")
    confidence_threshold: float = Field(
        default=0.5, ge=0, le=1, description="Confidence threshold"
    )
    model_profile: str = Field(
        default="balanced", description="Model profile (fast, balanced, high_quality)"
    )


class FaceDetectionRequest(BaseModel):
    """Request model for face detection inference."""

    video_path: str = Field(..., description="Path to video file")
    model_name: str = Field(default="yolov8n-face.pt", description="Model file name")
    frame_interval: int = Field(default=30, description="Process every Nth frame")
    confidence_threshold: float = Field(
        default=0.5, ge=0, le=1, description="Confidence threshold"
    )


class TranscriptionRequest(BaseModel):
    """Request model for transcription inference."""

    video_path: str = Field(..., description="Path to video file")
    model_name: str = Field(
        default="large-v3", description="Whisper model size (tiny, base, small, medium, large)"
    )
    language: str | None = Field(
        default=None, description="ISO 639-1 language code (null for auto-detect)"
    )
    vad_filter: bool = Field(default=True, description="Use VAD filter")


class OCRRequest(BaseModel):
    """Request model for OCR inference."""

    video_path: str = Field(..., description="Path to video file")
    frame_interval: int = Field(default=60, description="Process every Nth frame")
    languages: list[str] = Field(default=["en"], description="Languages to detect")
    use_gpu: bool = Field(default=True, description="Use GPU for inference")


class PlaceDetectionRequest(BaseModel):
    """Request model for place detection inference."""

    video_path: str = Field(..., description="Path to video file")
    frame_interval: int = Field(default=60, description="Process every Nth frame")
    top_k: int = Field(default=5, description="Top K predictions to return")


class SceneDetectionRequest(BaseModel):
    """Request model for scene detection inference."""

    video_path: str = Field(..., description="Path to video file")
    threshold: float = Field(default=0.4, ge=0, le=1, description="Scene threshold")
    min_scene_length: float = Field(
        default=0.6, description="Minimum scene length in seconds"
    )
