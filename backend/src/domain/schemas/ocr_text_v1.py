"""Schema for ocr.text artifact type version 1."""

from pydantic import BaseModel, Field


class PolygonPoint(BaseModel):
    """Point in a polygon bounding box."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class OcrTextV1(BaseModel):
    """
    Payload schema for ocr.text artifacts.

    Represents text detected and extracted from video frames with
    polygon bounding box and language metadata.
    """

    text: str = Field(..., description="Detected text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    polygon: list[PolygonPoint] = Field(
        ...,
        min_length=3,
        description="Polygon points defining text bounding box",
    )
    languages: list[str] = Field(
        ..., description="List of languages used for detection"
    )
    frame_index: int = Field(
        ..., ge=0, description="Frame number where text was detected"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Welcome to the presentation",
                    "confidence": 0.94,
                    "bounding_box": [
                        {"x": 100.0, "y": 50.0},
                        {"x": 400.0, "y": 50.0},
                        {"x": 400.0, "y": 100.0},
                        {"x": 100.0, "y": 100.0},
                    ],
                    "language": "en",
                    "frame_number": 180,
                }
            ]
        }
    }
