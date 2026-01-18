from ..domain.models import Transcription
from ..repositories.interfaces import TranscriptionRepository


class TranscriptionService:
    """Service layer for Transcription business operations."""

    def __init__(self, transcription_repository: TranscriptionRepository):
        self.transcription_repository = transcription_repository

    def save_transcription(self, transcription: Transcription) -> Transcription:
        """Save a transcription segment."""
        # Business logic: Validate time range
        if transcription.start >= transcription.end:
            raise ValueError("Start time must be before end time")

        if transcription.start < 0:
            raise ValueError("Start time cannot be negative")

        return self.transcription_repository.save(transcription)

    def get_video_transcriptions(self, video_id: str) -> list[Transcription]:
        """Get all transcriptions for a video, ordered by time."""
        return self.transcription_repository.find_by_video_id(video_id)

    def get_transcriptions_in_range(
        self, video_id: str, start: float, end: float
    ) -> list[Transcription]:
        """Get transcriptions within a specific time range."""
        if start >= end:
            raise ValueError("Start time must be before end time")

        return self.transcription_repository.find_by_time_range(video_id, start, end)

    def search_transcriptions(self, video_id: str, query: str) -> list[Transcription]:
        """Search transcriptions by text content."""
        all_transcriptions = self.transcription_repository.find_by_video_id(video_id)
        query_lower = query.lower()

        # Simple text search - could be enhanced with full-text search
        return [t for t in all_transcriptions if query_lower in t.text.lower()]

    def get_high_confidence_transcriptions(
        self, video_id: str, threshold: float = 0.8
    ) -> list[Transcription]:
        """Get transcriptions with confidence above threshold."""
        all_transcriptions = self.transcription_repository.find_by_video_id(video_id)
        return [t for t in all_transcriptions if t.is_high_confidence(threshold)]

    def delete_video_transcriptions(self, video_id: str) -> bool:
        """Delete all transcriptions for a video."""
        return self.transcription_repository.delete_by_video_id(video_id)
