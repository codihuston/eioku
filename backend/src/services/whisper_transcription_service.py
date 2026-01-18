"""Whisper transcription service for video audio."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing and metadata."""

    start: float
    end: float
    text: str
    confidence: float = 0.0
    speaker: str | None = None

    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start

    def is_low_confidence(self, threshold: float = 0.7) -> bool:
        """Check if segment has low confidence."""
        return self.confidence < threshold


@dataclass
class TranscriptionResult:
    """Complete transcription result for a video."""

    video_id: str
    segments: list[TranscriptionSegment]
    language: str
    duration: float
    model_name: str
    processing_time: float

    def get_full_text(self) -> str:
        """Get complete transcription text."""
        return " ".join(segment.text.strip() for segment in self.segments)

    def get_low_confidence_segments(
        self, threshold: float = 0.7
    ) -> list[TranscriptionSegment]:
        """Get segments with low confidence scores."""
        return [seg for seg in self.segments if seg.is_low_confidence(threshold)]

    def get_segments_in_range(
        self, start_time: float, end_time: float
    ) -> list[TranscriptionSegment]:
        """Get segments within a time range."""
        return [
            seg
            for seg in self.segments
            if seg.start >= start_time and seg.end <= end_time
        ]


class WhisperTranscriptionError(Exception):
    """Exception raised when transcription fails."""

    pass


class WhisperTranscriptionService:
    """Service for transcribing audio using OpenAI Whisper."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """Initialize Whisper transcription service.

        Args:
            model_name: Whisper model to use (large-v3, large-v3-turbo,
                       medium, small, base)
            device: Device to use (auto, cpu, cuda)
            compute_type: Compute precision (auto, float16, float32, int8)
                         'auto' will select the best option for the device
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = self._determine_compute_type(compute_type, device)
        self.model = None

        # Load model on first use (lazy loading)
        self._model_loaded = False

    def _determine_compute_type(self, compute_type: str, device: str) -> str:
        """Determine the best compute type for the device.

        Args:
            compute_type: Requested compute type or 'auto'
            device: Target device

        Returns:
            Appropriate compute type for the device
        """
        if compute_type != "auto":
            return compute_type

        # Auto-detect based on device
        if device == "cpu" or device == "auto":
            # CPU doesn't support float16 efficiently, use int8 for speed
            # or float32 for accuracy
            selected = "int8"
            logger.info(
                f"Auto-selected compute_type='{selected}' for device='{device}'"
            )
            return selected
        elif "cuda" in device.lower():
            # GPU supports float16 efficiently
            selected = "float16"
            logger.info(
                f"Auto-selected compute_type='{selected}' for device='{device}'"
            )
            return selected
        else:
            # Default to float32 for unknown devices
            selected = "float32"
            logger.info(
                f"Auto-selected compute_type='{selected}' for device='{device}'"
            )
            return selected

    def _load_model(self) -> None:
        """Load Whisper model (lazy loading)."""
        if self._model_loaded:
            return

        try:
            # Try faster-whisper first (more efficient)
            try:
                from faster_whisper import WhisperModel

                self.model = WhisperModel(
                    self.model_name, device=self.device, compute_type=self.compute_type
                )
                self._use_faster_whisper = True
                logger.info(f"Loaded faster-whisper model: {self.model_name}")
            except ImportError:
                # Fall back to openai-whisper
                import whisper

                self.model = whisper.load_model(self.model_name, device=self.device)
                self._use_faster_whisper = False
                logger.info(f"Loaded openai-whisper model: {self.model_name}")

            self._model_loaded = True

        except Exception as e:
            raise WhisperTranscriptionError(f"Failed to load Whisper model: {e}")

    def transcribe_audio(
        self, audio_path: str, video_id: str, language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to the audio file
            video_id: ID of the video being transcribed
            language: Language code (auto-detect if None)

        Returns:
            TranscriptionResult with segments and metadata

        Raises:
            WhisperTranscriptionError: If transcription fails
        """

        if not Path(audio_path).exists():
            raise WhisperTranscriptionError(f"Audio file not found: {audio_path}")

        self._load_model()

        start_time = time.time()

        try:
            if self._use_faster_whisper:
                result = self._transcribe_with_faster_whisper(audio_path, language)
            else:
                result = self._transcribe_with_openai_whisper(audio_path, language)

            processing_time = time.time() - start_time

            # Create transcription result
            transcription_result = TranscriptionResult(
                video_id=video_id,
                segments=result["segments"],
                language=result["language"],
                duration=result["duration"],
                model_name=self.model_name,
                processing_time=processing_time,
            )

            logger.info(
                f"Transcribed {len(transcription_result.segments)} segments "
                f"in {processing_time:.2f}s for video {video_id}"
            )

            return transcription_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Transcription failed after {processing_time:.2f}s: {e}"
            logger.error(error_msg)
            raise WhisperTranscriptionError(error_msg)

    def _transcribe_with_faster_whisper(
        self, audio_path: str, language: str | None
    ) -> dict[str, any]:
        """Transcribe using faster-whisper library."""
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Convert segments to our format
        transcription_segments = []
        for segment in segments:
            transcription_segments.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=segment.avg_logprob
                    if hasattr(segment, "avg_logprob")
                    else 0.0,
                )
            )

        return {
            "segments": transcription_segments,
            "language": info.language,
            "duration": info.duration,
        }

    def _transcribe_with_openai_whisper(
        self, audio_path: str, language: str | None
    ) -> dict[str, any]:
        """Transcribe using openai-whisper library."""
        result = self.model.transcribe(
            audio_path, language=language, word_timestamps=True, verbose=False
        )

        # Convert segments to our format
        transcription_segments = []
        for segment in result["segments"]:
            transcription_segments.append(
                TranscriptionSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("avg_logprob", 0.0),
                )
            )

        return {
            "segments": transcription_segments,
            "language": result["language"],
            "duration": result.get("duration", 0.0),
        }

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
            "ar",
            "hi",
            "tr",
            "pl",
            "nl",
            "sv",
            "da",
            "no",
            "fi",
        ]

    def estimate_processing_time(self, duration_seconds: float) -> float:
        """Estimate transcription processing time."""
        model_multipliers = {
            "large-v3": 0.3,
            "large-v3-turbo": 0.2,
            "medium": 0.15,
            "small": 0.1,
            "base": 0.05,
        }

        multiplier = model_multipliers.get(self.model_name, 0.3)

        if self.device == "cpu":
            multiplier *= 3

        return duration_seconds * multiplier
