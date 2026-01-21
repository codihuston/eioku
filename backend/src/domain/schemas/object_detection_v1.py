"""Schema for object.detection artifact type version 1."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects."""

    x: float = Field(..., ge=0.0, description="X coordinate (top-left)")
    y: float = Field(..., ge=0.0, description="Y coordinate (top-left)")
    width: float = Field(..., gt=0.0, description="Width of bounding box")
    height: float = Field(..., gt=0.0, description="Height of bounding box")


class ObjectDetectionV1(BaseModel):
    """
    Payload schema for object.detection artifacts.

    Represents a detected object in video frames with bounding box,
    label, and confidence.
    """

    label: str = Field(..., description="Object class label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    bounding_box: BoundingBox = Field(..., description="Object bounding box")
    frame_number: int = Field(
        ..., ge=0, description="Frame number where object was detected"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "person",
                    "confidence": 0.92,
                    "bounding_box": {
                        "x": 100.0,
                        "y": 150.0,
                        "width": 200.0,
                        "height": 300.0,
                    },
                    "frame_number": 450,
                }
            ]
        }
    }
