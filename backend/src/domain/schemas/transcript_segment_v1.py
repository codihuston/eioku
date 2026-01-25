"""Schema for transcription artifact type version 1."""

from pydantic import BaseModel, Field


class Word(BaseModel):
    """Word-level transcription detail."""

    word: str = Field(..., description="The word text")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Word confidence score"
    )


class TranscriptSegmentV1(BaseModel):
    """
    Payload schema for transcription artifacts.

    Represents a transcribed segment of audio with text, timing,
    confidence, and optional word-level details.
    """

    text: str = Field(..., description="Transcribed text")
    start_ms: int = Field(..., ge=0, description="Start time in milliseconds")
    end_ms: int = Field(..., ge=0, description="End time in milliseconds")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Segment confidence score"
    )
    words: list[Word] | None = Field(
        default=None, description="Word-level transcription details"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hello, how are you today?",
                    "start_ms": 1000,
                    "end_ms": 3500,
                    "confidence": 0.98,
                    "words": [
                        {
                            "word": "Hello",
                            "start": 1.0,
                            "end": 1.3,
                            "confidence": 0.99,
                        },
                        {
                            "word": "how",
                            "start": 1.4,
                            "end": 1.6,
                            "confidence": 0.98,
                        },
                    ],
                }
            ]
        }
    }
