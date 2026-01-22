"""Transcription task handler for video processing orchestration."""

import hashlib
import json
import logging
import uuid
from datetime import datetime

from ..domain.artifacts import ArtifactEnvelope
from ..domain.models import Task, Video
from ..domain.schema_registry import SchemaRegistry
from ..domain.schemas.transcript_segment_v1 import TranscriptSegmentV1
from ..repositories.interfaces import ArtifactRepository
from .audio_extraction_service import AudioExtractionService
from .whisper_transcription_service import WhisperTranscriptionService

logger = logging.getLogger(__name__)


class TranscriptionTaskHandler:
    """Handles transcription tasks in the orchestration system."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        schema_registry: SchemaRegistry,
        audio_service: AudioExtractionService | None = None,
        whisper_service: WhisperTranscriptionService | None = None,
    ):
        self.artifact_repository = artifact_repository
        self.schema_registry = schema_registry
        self.audio_service = audio_service or AudioExtractionService()
        self.whisper_service = whisper_service or WhisperTranscriptionService(
            model_name="base", device="cpu", compute_type="float32"
        )

    def _compute_config_hash(self, config: dict) -> str:
        """Compute hash of configuration for provenance tracking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _compute_input_hash(self, video_path: str) -> str:
        """Compute hash of input video file for provenance tracking."""
        # For now, use video path as input identifier
        # In production, could use file hash or video_id
        return hashlib.sha256(video_path.encode()).hexdigest()[:16]

    def process_transcription_task(
        self,
        task: Task,
        video: Video,
        run_id: str | None = None,
        model_profile: str | None = None,
    ) -> bool:
        """Process a transcription task for a video.

        Args:
            task: The transcription task to process
            video: The video to transcribe
            run_id: Optional run ID for tracking (generated if not provided)
            model_profile: Optional model profile (fast, balanced, high_quality).
                          If not provided, determined from model name.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting transcription for video {video.video_id}")

            # Generate run_id if not provided
            if run_id is None:
                run_id = str(uuid.uuid4())
                logger.info(f"Generated run_id: {run_id}")

            # Step 1: Extract audio from video
            logger.info(f"Extracting audio from {video.file_path}")
            audio_path = self.audio_service.extract_audio(video.file_path)

            # Step 2: Transcribe audio using Whisper
            logger.info("Transcribing audio with Whisper")
            transcription_result = self.whisper_service.transcribe_audio(
                audio_path, video.video_id
            )

            # Compute provenance hashes
            config = {
                "model_name": self.whisper_service.model_name,
                "device": self.whisper_service.device,
                "compute_type": self.whisper_service.compute_type,
            }
            config_hash = self._compute_config_hash(config)
            input_hash = self._compute_input_hash(video.file_path)

            # Determine model profile - use provided or infer from model name
            if model_profile is None:
                model_profile = self._determine_model_profile(
                    self.whisper_service.model_name
                )

            # Step 3: Save transcription segments as artifacts
            logger.info(
                f"Saving {len(transcription_result.segments)} segments as artifacts"
            )
            saved_count = 0

            for segment in transcription_result.segments:
                # Create payload using Pydantic schema
                payload = TranscriptSegmentV1(
                    text=segment.text,
                    speaker=segment.speaker,
                    confidence=segment.confidence,
                    language=transcription_result.language,
                )

                # Create artifact envelope
                artifact = ArtifactEnvelope(
                    artifact_id=str(uuid.uuid4()),
                    asset_id=video.video_id,
                    artifact_type="transcript.segment",
                    schema_version=1,
                    span_start_ms=int(segment.start * 1000),  # Convert to milliseconds
                    span_end_ms=int(segment.end * 1000),  # Convert to milliseconds
                    payload_json=payload.model_dump_json(),
                    producer="whisper",
                    producer_version=self.whisper_service.model_name,
                    model_profile=model_profile,
                    config_hash=config_hash,
                    input_hash=input_hash,
                    run_id=run_id,
                    created_at=datetime.utcnow(),
                )

                # Save to artifact repository
                self.artifact_repository.create(artifact)
                saved_count += 1

            # Step 4: Clean up temporary audio file
            try:
                self.audio_service.cleanup_audio_file(audio_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup audio file {audio_path}: {e}")

            logger.info(
                f"Transcription completed for video {video.video_id}: "
                f"{saved_count} artifacts saved, "
                f"processing time: {transcription_result.processing_time:.1f}s"
            )

            return True

        except Exception as e:
            logger.error(f"Transcription failed for video {video.video_id}: {e}")
            return False

    def _determine_model_profile(self, model_name: str) -> str:
        """Determine model profile based on model name."""
        if "large" in model_name:
            return "high_quality"
        elif "medium" in model_name:
            return "balanced"
        else:
            return "fast"

    def get_transcription_segments(self, video_id: str) -> list[ArtifactEnvelope]:
        """Get all transcription segments for a video."""
        return self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="transcript.segment"
        )

    def get_transcription_text(self, video_id: str) -> str:
        """Get full transcription text for a video."""
        artifacts = self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="transcript.segment"
        )

        # Parse payloads and extract text
        texts = []
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            texts.append(payload["text"])

        return " ".join(texts)

    def search_transcription(self, video_id: str, query: str) -> list[ArtifactEnvelope]:
        """Search transcription segments by text."""
        artifacts = self.artifact_repository.get_by_asset(
            asset_id=video_id, artifact_type="transcript.segment"
        )

        query_lower = query.lower()
        matching_artifacts = []

        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            if query_lower in payload["text"].lower():
                matching_artifacts.append(artifact)

        return matching_artifacts
