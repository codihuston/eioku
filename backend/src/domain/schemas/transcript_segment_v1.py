"""Schema for transcript.segment artifact type version 1."""

from pydantic import BaseModel, Field


class TranscriptSegmentV1(BaseModel):
    """
    Payload schema for transcript.segment artifacts.

    Represents a time-aligned piece of transcribed text with speaker
    and confidence information.
    """

    text: str = Field(..., description="Transcribed text content")
    speaker: str | None = Field(None, description="Speaker identifier or name")
    confidence: float = Field(
        ...,
        le=1.0,
        description="Transcription confidence score (log probability, can be negative)",
    )
    language: str = Field(default="en", description="Language code (ISO 639-1)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hello, welcome to the presentation.",
                    "speaker": "Speaker 1",
                    "confidence": 0.95,
                    "language": "en",
                }
            ]
        }
    }
