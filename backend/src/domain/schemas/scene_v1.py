"""Schema for scene.detection artifact type version 1."""

from pydantic import BaseModel, Field


class SceneV1(BaseModel):
    """
    Payload schema for scene detection artifacts.

    Represents detected scene boundaries in video with timing information.
    """

    scene_index: int = Field(..., ge=0, description="Scene index (0-based)")
    start_ms: int = Field(..., ge=0, description="Scene start time in milliseconds")
    end_ms: int = Field(..., ge=0, description="Scene end time in milliseconds")
    duration_ms: int = Field(..., gt=0, description="Scene duration in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scene_index": 0,
                    "start_ms": 0,
                    "end_ms": 5000,
                    "duration_ms": 5000,
                },
                {
                    "scene_index": 1,
                    "start_ms": 5000,
                    "end_ms": 12500,
                    "duration_ms": 7500,
                },
            ]
        }
    }
