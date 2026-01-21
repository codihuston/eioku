"""Schema for face.detection artifact type version 1."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected faces."""

    x: float = Field(..., ge=0.0, description="X coordinate (top-left)")
    y: float = Field(..., ge=0.0, description="Y coordinate (top-left)")
    width: float = Field(..., gt=0.0, description="Width of bounding box")
    height: float = Field(..., gt=0.0, description="Height of bounding box")


class FaceDetectionV1(BaseModel):
    """
    Payload schema for face.detection artifacts.

    Represents a detected face in video frames with bounding box
    and optional cluster assignment.
    """

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    bounding_box: BoundingBox = Field(..., description="Face bounding box")
    cluster_id: str | None = Field(
        None, description="Face cluster identifier for grouping"
    )
    frame_number: int = Field(
        ..., ge=0, description="Frame number where face was detected"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "confidence": 0.98,
                    "bounding_box": {
                        "x": 250.0,
                        "y": 100.0,
                        "width": 150.0,
                        "height": 180.0,
                    },
                    "cluster_id": "person_001",
                    "frame_number": 320,
                }
            ]
        }
    }
