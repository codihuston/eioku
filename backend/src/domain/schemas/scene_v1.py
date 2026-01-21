"""Schema for scene artifact type version 1."""

from pydantic import BaseModel, Field


class SceneV1(BaseModel):
    """
    Payload schema for scene artifacts.

    Represents a continuous video segment detected by scene change
    detection algorithms.
    """

    scene_index: int = Field(..., ge=0, description="Sequential scene number")
    method: str = Field(
        ..., description="Detection method used (e.g., 'content', 'threshold')"
    )
    score: float = Field(..., ge=0.0, description="Scene change detection score")
    frame_number: int = Field(..., ge=0, description="Frame number where scene starts")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scene_index": 0,
                    "method": "content",
                    "score": 0.85,
                    "frame_number": 120,
                }
            ]
        }
    }
