"""Tests for Whisper transcription service."""

from unittest.mock import patch

import pytest

from src.services.whisper_transcription_service import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperTranscriptionError,
    WhisperTranscriptionService,
)


class TestTranscriptionSegment:
    """Test TranscriptionSegment dataclass."""

    def test_duration_calculation(self):
        """Test segment duration calculation."""
        segment = TranscriptionSegment(start=10.5, end=15.2, text="Hello world")
        assert abs(segment.duration() - 4.7) < 0.001

    def test_low_confidence_detection(self):
        """Test low confidence detection."""
        high_conf = TranscriptionSegment(
            start=0.0, end=5.0, text="Clear speech", confidence=0.9
        )
        assert not high_conf.is_low_confidence()

        low_conf = TranscriptionSegment(
            start=0.0, end=5.0, text="Unclear speech", confidence=0.5
        )
        assert low_conf.is_low_confidence()


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def setup_method(self):
        """Set up test data."""
        self.segments = [
            TranscriptionSegment(0.0, 5.0, "Hello world", 0.9),
            TranscriptionSegment(5.0, 10.0, "How are you", 0.8),
            TranscriptionSegment(10.0, 15.0, "I am fine", 0.6),
            TranscriptionSegment(15.0, 20.0, "Thank you", 0.95),
        ]

        self.result = TranscriptionResult(
            video_id="test_video",
            segments=self.segments,
            language="en",
            duration=20.0,
            model_name="large-v3",
            processing_time=5.2,
        )

    def test_get_full_text(self):
        """Test full text extraction."""
        expected = "Hello world How are you I am fine Thank you"
        assert self.result.get_full_text() == expected

    def test_get_low_confidence_segments(self):
        """Test low confidence segment filtering."""
        low_conf = self.result.get_low_confidence_segments(threshold=0.7)
        assert len(low_conf) == 1
        assert low_conf[0].text == "I am fine"

    def test_get_segments_in_range(self):
        """Test segment range filtering."""
        range_segments = self.result.get_segments_in_range(4.0, 12.0)
        assert len(range_segments) == 1
        assert range_segments[0].text == "How are you"


class TestWhisperTranscriptionService:
    """Test WhisperTranscriptionService."""

    def setup_method(self):
        """Set up test service."""
        self.service = WhisperTranscriptionService(model_name="base", device="cpu")

    def test_initialization(self):
        """Test service initialization."""
        assert self.service.model_name == "base"
        assert self.service.device == "cpu"
        assert not self.service._model_loaded

    @patch("src.services.whisper_transcription_service.Path")
    def test_transcribe_missing_file(self, mock_path):
        """Test transcription with missing audio file."""
        mock_path.return_value.exists.return_value = False

        with pytest.raises(WhisperTranscriptionError, match="Audio file not found"):
            self.service.transcribe_audio("missing.wav", "test_video")

    @patch("src.services.whisper_transcription_service.Path")
    @patch.object(WhisperTranscriptionService, "_load_model")
    @patch.object(WhisperTranscriptionService, "_transcribe_with_faster_whisper")
    def test_transcribe_with_faster_whisper(
        self, mock_transcribe, mock_load, mock_path
    ):
        """Test transcription using faster-whisper."""
        mock_path.return_value.exists.return_value = True
        self.service._model_loaded = True
        self.service._use_faster_whisper = True

        mock_segments = [
            TranscriptionSegment(0.0, 5.0, "Hello world", 0.9),
            TranscriptionSegment(5.0, 10.0, "How are you", 0.8),
        ]

        mock_transcribe.return_value = {
            "segments": mock_segments,
            "language": "en",
            "duration": 10.0,
        }

        result = self.service.transcribe_audio("test.wav", "video_123")

        assert result.video_id == "video_123"
        assert len(result.segments) == 2
        assert result.language == "en"

    def test_get_supported_languages(self):
        """Test supported languages list."""
        languages = self.service.get_supported_languages()

        assert isinstance(languages, list)
        assert "en" in languages
        assert "es" in languages

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        base_service = WhisperTranscriptionService(model_name="base", device="cuda")
        large_service = WhisperTranscriptionService(
            model_name="large-v3", device="cuda"
        )

        duration = 60.0

        base_time = base_service.estimate_processing_time(duration)
        large_time = large_service.estimate_processing_time(duration)

        assert base_time < large_time
        assert base_time > 0
