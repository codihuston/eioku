"""Schema for ocr artifact type version 1."""

from pydantic import BaseModel, Field


class Point(BaseModel):
    """2D point coordinates."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class OCRDetectionV1(BaseModel):
    """
    Payload schema for OCR artifacts.

    Represents detected text in video frames with polygon coordinates,
    text content, and confidence score.
    """

    text: str = Field(..., description="Detected text content")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    polygon: list[Point] = Field(
        ..., min_length=3, description="Text bounding polygon (at least 3 points)"
    )
    frame_number: int = Field(
        ..., ge=0, description="Frame number where text was detected"
    )
    language: str | None = Field(
        default=None, description="Detected language code (ISO 639-1)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "STOP",
                    "confidence": 0.92,
                    "polygon": [
                        {"x": 100.0, "y": 50.0},
                        {"x": 200.0, "y": 50.0},
                        {"x": 200.0, "y": 100.0},
                        {"x": 100.0, "y": 100.0},
                    ],
                    "frame_number": 450,
                    "language": "en",
                }
            ]
        }
    }
