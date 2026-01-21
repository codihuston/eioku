"""Schema for place.classification artifact type version 1."""

from pydantic import BaseModel, Field


class AlternativeLabel(BaseModel):
    """Alternative place classification with confidence."""

    label: str = Field(..., description="Alternative place label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class PlaceClassificationV1(BaseModel):
    """
    Payload schema for place.classification artifacts.

    Represents a scene/location classification (e.g., "kitchen", "office", "beach")
    with alternative labels and confidence scores.
    """

    label: str = Field(..., description="Primary place classification label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Primary classification confidence"
    )
    alternative_labels: list[AlternativeLabel] = Field(
        default_factory=list,
        description="Alternative classifications with confidence scores",
    )
    frame_number: int = Field(
        ..., ge=0, description="Frame number where classification was made"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "office",
                    "confidence": 0.87,
                    "alternative_labels": [
                        {"label": "conference_room", "confidence": 0.65},
                        {"label": "classroom", "confidence": 0.42},
                    ],
                    "frame_number": 600,
                }
            ]
        }
    }
