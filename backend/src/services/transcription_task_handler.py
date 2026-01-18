"""Transcription task handler for video processing orchestration."""

import logging
import uuid
from datetime import datetime

from ..domain.models import Task, Transcription, Video
from ..repositories.interfaces import TranscriptionRepository
from .audio_extraction_service import AudioExtractionService
from .whisper_transcription_service import WhisperTranscriptionService

logger = logging.getLogger(__name__)


class TranscriptionTaskHandler:
    """Handles transcription tasks in the orchestration system."""

    def __init__(
        self,
        transcription_repository: TranscriptionRepository,
        audio_service: AudioExtractionService | None = None,
        whisper_service: WhisperTranscriptionService | None = None,
    ):
        self.transcription_repository = transcription_repository
        self.audio_service = audio_service or AudioExtractionService()
        self.whisper_service = whisper_service or WhisperTranscriptionService(
            model_name="base", device="cpu", compute_type="float32"
        )

    def process_transcription_task(self, task: Task, video: Video) -> bool:
        """Process a transcription task for a video.

        Args:
            task: The transcription task to process
            video: The video to transcribe

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting transcription for video {video.video_id}")

            # Step 1: Extract audio from video
            logger.info(f"Extracting audio from {video.file_path}")
            audio_path = self.audio_service.extract_audio(video.file_path)

            # Step 2: Transcribe audio using Whisper
            logger.info("Transcribing audio with Whisper")
            transcription_result = self.whisper_service.transcribe_audio(
                audio_path, video.video_id
            )

            # Step 3: Save transcription segments to database
            logger.info(
                f"Saving {len(transcription_result.segments)} segments to database"
            )
            saved_count = 0

            for segment in transcription_result.segments:
                # Create domain model from transcription segment
                transcription = Transcription(
                    segment_id=str(uuid.uuid4()),
                    video_id=video.video_id,
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.confidence,
                    speaker=segment.speaker,
                    created_at=datetime.utcnow(),
                )

                # Save to database
                self.transcription_repository.save(transcription)
                saved_count += 1

            # Step 4: Clean up temporary audio file
            try:
                self.audio_service.cleanup_audio_file(audio_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup audio file {audio_path}: {e}")

            logger.info(
                f"Transcription completed for video {video.video_id}: "
                f"{saved_count} segments saved, "
                f"processing time: {transcription_result.processing_time:.1f}s"
            )

            return True

        except Exception as e:
            logger.error(f"Transcription failed for video {video.video_id}: {e}")
            return False

    def get_transcription_segments(self, video_id: str) -> list[Transcription]:
        """Get all transcription segments for a video."""
        return self.transcription_repository.find_by_video_id(video_id)

    def get_transcription_text(self, video_id: str) -> str:
        """Get full transcription text for a video."""
        segments = self.transcription_repository.find_by_video_id(video_id)
        return " ".join(segment.text for segment in segments)

    def search_transcription(self, video_id: str, query: str) -> list[Transcription]:
        """Search transcription segments by text."""
        segments = self.transcription_repository.find_by_video_id(video_id)
        query_lower = query.lower()

        return [segment for segment in segments if query_lower in segment.text.lower()]
