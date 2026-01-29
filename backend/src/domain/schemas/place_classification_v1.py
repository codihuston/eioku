"""Schema for place.classification artifact type version 1."""

from pydantic import BaseModel, Field


class PlacePrediction(BaseModel):
    """Place classification prediction."""

    label: str = Field(..., description="Place category label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence score"
    )


class PlaceClassificationV1(BaseModel):
    """
    Payload schema for place classification artifacts.

    Represents detected place/scene categories in video frames with
    top-k predictions and confidence scores.
    """

    predictions: list[PlacePrediction] = Field(
        ..., min_length=1, description="Top place predictions"
    )
    frame_number: int = Field(
        ..., ge=0, description="Frame number where classification was performed"
    )
    top_k: int = Field(
        default=5, ge=1, description="Number of top predictions returned"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predictions": [
                        {"label": "beach", "confidence": 0.85},
                        {"label": "coast", "confidence": 0.12},
                        {"label": "ocean", "confidence": 0.02},
                    ],
                    "frame_number": 600,
                    "top_k": 3,
                }
            ]
        }
    }
