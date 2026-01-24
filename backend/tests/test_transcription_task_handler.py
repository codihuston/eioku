"""Tests for transcription task handler."""

from datetime import datetime
from unittest.mock import Mock

from src.domain.artifacts import ArtifactEnvelope
from src.domain.models import Task, Video
from src.services.transcription_task_handler import TranscriptionTaskHandler
from src.services.whisper_transcription_service import (
    TranscriptionResult,
    TranscriptionSegment,
)


class TestTranscriptionTaskHandler:
    """Test TranscriptionTaskHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_artifact_repo = Mock()
        self.mock_schema_registry = Mock()
        self.mock_audio_service = Mock()
        self.mock_whisper_service = Mock()

        self.handler = TranscriptionTaskHandler(
            artifact_repository=self.mock_artifact_repo,
            schema_registry=self.mock_schema_registry,
            audio_service=self.mock_audio_service,
            whisper_service=self.mock_whisper_service,
        )

        # Test data
        self.video = Video(
            video_id="test_video_123",
            file_path="/test/video.mkv",
            filename="video.mkv",
            last_modified=datetime.utcnow(),
        )

        self.task = Task(
            task_id="task_123",
            video_id="test_video_123",
            task_type="transcription",
            status="pending",
            created_at=datetime.utcnow(),
        )

    def test_process_transcription_task_success(self):
        """Test successful transcription processing."""
        # Mock audio extraction
        self.mock_audio_service.extract_audio.return_value = "/tmp/audio.wav"

        # Mock transcription result
        segments = [
            TranscriptionSegment(0.0, 5.0, "Hello world", 0.9),
            TranscriptionSegment(5.0, 10.0, "How are you", 0.8),
        ]

        transcription_result = TranscriptionResult(
            video_id="test_video_123",
            segments=segments,
            language="en",
            duration=10.0,
            model_name="base",
            processing_time=2.5,
        )

        self.mock_whisper_service.transcribe_audio.return_value = transcription_result
        self.mock_whisper_service.model_name = "base"
        self.mock_whisper_service.device = "cpu"
        self.mock_whisper_service.compute_type = "float32"

        # Mock artifact repository batch_create
        self.mock_artifact_repo.batch_create.return_value = []

        # Process task
        result = self.handler.process_transcription_task(self.task, self.video)

        # Verify success
        assert result is True

        # Verify audio extraction called
        self.mock_audio_service.extract_audio.assert_called_once_with("/test/video.mkv")

        # Verify transcription called
        self.mock_whisper_service.transcribe_audio.assert_called_once_with(
            "/tmp/audio.wav", "test_video_123"
        )

        # Verify batch_create called once with 2 artifacts
        self.mock_artifact_repo.batch_create.assert_called_once()
        call_args = self.mock_artifact_repo.batch_create.call_args
        artifacts = call_args[0][0]  # First positional argument
        assert len(artifacts) == 2

        # Verify cleanup called
        self.mock_audio_service.cleanup_audio_file.assert_called_once_with(
            "/tmp/audio.wav"
        )

    def test_process_transcription_task_audio_extraction_failure(self):
        """Test transcription failure during audio extraction."""
        # Mock audio extraction failure
        self.mock_audio_service.extract_audio.side_effect = Exception(
            "Audio extraction failed"
        )

        # Process task
        result = self.handler.process_transcription_task(self.task, self.video)

        # Verify failure
        assert result is False

        # Verify transcription not called
        self.mock_whisper_service.transcribe_audio.assert_not_called()

        # Verify no artifacts created
        self.mock_artifact_repo.create.assert_not_called()

    def test_process_transcription_task_whisper_failure(self):
        """Test transcription failure during Whisper processing."""
        # Mock audio extraction success
        self.mock_audio_service.extract_audio.return_value = "/tmp/audio.wav"

        # Mock Whisper failure
        self.mock_whisper_service.transcribe_audio.side_effect = Exception(
            "Whisper failed"
        )

        # Process task
        result = self.handler.process_transcription_task(self.task, self.video)

        # Verify failure
        assert result is False

        # Verify no artifacts created
        self.mock_artifact_repo.create.assert_not_called()

    def test_get_transcription_segments(self):
        """Test getting transcription segments."""
        # Mock artifact repository response
        mock_artifacts = [Mock(spec=ArtifactEnvelope), Mock(spec=ArtifactEnvelope)]
        self.mock_artifact_repo.get_by_asset.return_value = mock_artifacts

        # Get segments
        result = self.handler.get_transcription_segments("test_video")

        # Verify
        assert result == mock_artifacts
        self.mock_artifact_repo.get_by_asset.assert_called_once_with(
            asset_id="test_video", artifact_type="transcript.segment"
        )

    def test_get_transcription_text(self):
        """Test getting full transcription text."""
        # Mock artifacts with text payloads
        mock_artifacts = [
            Mock(
                payload_json=(
                    '{"text": "Hello world", ' '"confidence": 0.9, "language": "en"}'
                )
            ),
            Mock(
                payload_json=(
                    '{"text": "How are you", ' '"confidence": 0.8, "language": "en"}'
                )
            ),
            Mock(
                payload_json=(
                    '{"text": "I am fine", ' '"confidence": 0.85, "language": "en"}'
                )
            ),
        ]
        self.mock_artifact_repo.get_by_asset.return_value = mock_artifacts

        # Get text
        result = self.handler.get_transcription_text("test_video")

        # Verify
        assert result == "Hello world How are you I am fine"

    def test_search_transcription(self):
        """Test searching transcription segments."""
        # Mock artifacts
        mock_artifacts = [
            Mock(
                payload_json=(
                    '{"text": "Hello world", ' '"confidence": 0.9, "language": "en"}'
                )
            ),
            Mock(
                payload_json=(
                    '{"text": "How are you", ' '"confidence": 0.8, "language": "en"}'
                )
            ),
            Mock(
                payload_json=(
                    '{"text": "Goodbye world", ' '"confidence": 0.85, "language": "en"}'
                )
            ),
        ]
        self.mock_artifact_repo.get_by_asset.return_value = mock_artifacts

        # Search for "world"
        result = self.handler.search_transcription("test_video", "world")

        # Verify - should find 2 artifacts containing "world"
        assert len(result) == 2

    def test_cleanup_failure_does_not_affect_success(self):
        """Test that cleanup failure doesn't affect overall success."""
        # Mock successful processing
        self.mock_audio_service.extract_audio.return_value = "/tmp/audio.wav"

        segments = [TranscriptionSegment(0.0, 5.0, "Test", 0.9)]
        transcription_result = TranscriptionResult(
            video_id="test_video_123",
            segments=segments,
            language="en",
            duration=5.0,
            model_name="base",
            processing_time=1.0,
        )

        self.mock_whisper_service.transcribe_audio.return_value = transcription_result
        self.mock_whisper_service.model_name = "base"
        self.mock_whisper_service.device = "cpu"
        self.mock_whisper_service.compute_type = "float32"
        self.mock_artifact_repo.batch_create.return_value = []

        # Mock cleanup failure
        self.mock_audio_service.cleanup_audio_file.side_effect = Exception(
            "Cleanup failed"
        )

        # Process task
        result = self.handler.process_transcription_task(self.task, self.video)

        # Should still succeed despite cleanup failure
        assert result is True
        self.mock_artifact_repo.batch_create.assert_called_once()

    def test_determine_model_profile(self):
        """Test model profile determination."""
        assert self.handler._determine_model_profile("large-v3") == "high_quality"
        assert self.handler._determine_model_profile("large-v3-turbo") == "high_quality"
        assert self.handler._determine_model_profile("medium") == "balanced"
        assert self.handler._determine_model_profile("small") == "fast"
        assert self.handler._determine_model_profile("base") == "fast"
