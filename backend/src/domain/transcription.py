from datetime import datetime


class Transcription:
    """Domain model for Transcription - pure business object."""

    def __init__(
        self,
        segment_id: str,
        video_id: str,
        text: str,
        start: float,
        end: float,
        confidence: float | None = None,
        speaker: str | None = None,
        created_at: datetime | None = None,
    ):
        self.segment_id = segment_id
        self.video_id = video_id
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
        self.speaker = speaker
        self.created_at = created_at

    def get_duration(self) -> float:
        """Get duration of transcription segment in seconds."""
        return self.end - self.start

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if transcription confidence is above threshold."""
        return self.confidence is not None and self.confidence >= threshold
