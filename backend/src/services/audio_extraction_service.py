"""Audio extraction service for video transcription."""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Exception raised when audio extraction fails."""
    pass


class AudioExtractionService:
    """Service for extracting audio from video files using FFmpeg."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize audio extraction service.
        
        Args:
            temp_dir: Directory for temporary audio files. Uses system temp if None.
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
        
        # Verify FFmpeg is available
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is installed and accessible."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise AudioExtractionError("FFmpeg is not working properly")
            logger.info("FFmpeg verification successful")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise AudioExtractionError(f"FFmpeg not found or not working: {e}")
    
    def extract_audio(self, video_path: str, audio_format: str = "wav") -> str:
        """Extract audio from video file.
        
        Args:
            video_path: Path to the input video file
            audio_format: Output audio format (wav, mp3, flac)
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            AudioExtractionError: If extraction fails
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")
        
        # Create temporary audio file
        audio_filename = f"{video_path.stem}_{os.getpid()}.{audio_format}"
        audio_path = self.temp_dir / audio_filename
        
        try:
            # FFmpeg command to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", self._get_audio_codec(audio_format),
                "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
                "-ac", "1",  # Mono channel
                "-y",  # Overwrite output file
                str(audio_path)
            ]
            
            logger.info(f"Extracting audio from {video_path} to {audio_path}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"FFmpeg failed: {result.stderr}"
                logger.error(error_msg)
                raise AudioExtractionError(error_msg)
            
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise AudioExtractionError("Audio extraction produced empty file")
            
            logger.info(f"Audio extracted successfully: {audio_path}")
            return str(audio_path)
            
        except subprocess.TimeoutExpired:
            error_msg = f"Audio extraction timed out for {video_path}"
            logger.error(error_msg)
            # Clean up partial file
            if audio_path.exists():
                audio_path.unlink()
            raise AudioExtractionError(error_msg)
        
        except Exception as e:
            # Clean up on any error
            if audio_path.exists():
                audio_path.unlink()
            raise AudioExtractionError(f"Audio extraction failed: {e}")
    
    def _get_audio_codec(self, audio_format: str) -> str:
        """Get appropriate audio codec for format."""
        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "libmp3lame", 
            "flac": "flac"
        }
        return codec_map.get(audio_format.lower(), "pcm_s16le")
    
    def cleanup_audio_file(self, audio_path: str) -> None:
        """Clean up temporary audio file.
        
        Args:
            audio_path: Path to the audio file to delete
        """
        try:
            audio_file = Path(audio_path)
            if audio_file.exists():
                audio_file.unlink()
                logger.info(f"Cleaned up audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio file {audio_path}: {e}")
    
    def get_audio_info(self, video_path: str) -> dict:
        """Get audio information from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with audio information (duration, sample_rate, channels)
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a:0",  # First audio stream
                str(video_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise AudioExtractionError(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            if not data.get("streams"):
                raise AudioExtractionError("No audio streams found in video")
            
            stream = data["streams"][0]
            
            return {
                "duration": float(stream.get("duration", 0)),
                "sample_rate": int(stream.get("sample_rate", 0)),
                "channels": int(stream.get("channels", 0)),
                "codec": stream.get("codec_name", "unknown")
            }
            
        except json.JSONDecodeError as e:
            raise AudioExtractionError(f"Failed to parse FFprobe output: {e}")
        except subprocess.TimeoutExpired:
            raise AudioExtractionError("FFprobe timed out")
        except Exception as e:
            raise AudioExtractionError(f"Failed to get audio info: {e}")
