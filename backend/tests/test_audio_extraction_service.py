"""Tests for audio extraction service."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.services.audio_extraction_service import AudioExtractionService, AudioExtractionError


class TestAudioExtractionService:
    """Test audio extraction service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_service_initialization(self):
        """Test service initialization."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            assert service.temp_dir == Path(self.temp_dir)
    
    def test_ffmpeg_verification_success(self):
        """Test successful FFmpeg verification."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            # Should not raise exception
            service._verify_ffmpeg()
    
    def test_ffmpeg_verification_failure(self):
        """Test FFmpeg verification failure."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")
            
            with pytest.raises(AudioExtractionError, match="FFmpeg not found"):
                AudioExtractionService(self.temp_dir)
    
    def test_extract_audio_success(self):
        """Test successful audio extraction."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            # Mock FFmpeg verification
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            # Create fake video file
            video_path = Path(self.temp_dir) / "test_video.mp4"
            video_path.write_text("fake video content")
            
            # Mock successful extraction
            def mock_extraction(*args, **kwargs):
                # Create fake audio file
                if "ffmpeg" in args[0][0]:
                    audio_path = Path(args[0][-1])  # Last argument is output path
                    audio_path.write_text("fake audio content")
                return Mock(returncode=0, stderr="")
            
            mock_run.side_effect = mock_extraction
            
            # Extract audio
            audio_path = service.extract_audio(str(video_path))
            
            assert Path(audio_path).exists()
            assert audio_path.endswith(".wav")
    
    def test_extract_audio_video_not_found(self):
        """Test audio extraction with missing video file."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            with pytest.raises(AudioExtractionError, match="Video file not found"):
                service.extract_audio("/nonexistent/video.mp4")
    
    def test_extract_audio_ffmpeg_failure(self):
        """Test audio extraction with FFmpeg failure."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            # Mock FFmpeg verification success
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            # Create fake video file
            video_path = Path(self.temp_dir) / "test_video.mp4"
            video_path.write_text("fake video content")
            
            # Mock FFmpeg failure
            def mock_extraction(*args, **kwargs):
                if "ffmpeg" in args[0][0] and len(args[0]) > 5:  # Extraction command
                    return Mock(returncode=1, stderr="FFmpeg error")
                return Mock(returncode=0, stderr="")
            
            mock_run.side_effect = mock_extraction
            
            with pytest.raises(AudioExtractionError, match="FFmpeg failed"):
                service.extract_audio(str(video_path))
    
    def test_extract_audio_timeout(self):
        """Test audio extraction timeout."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            # Mock FFmpeg verification success
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            # Create fake video file
            video_path = Path(self.temp_dir) / "test_video.mp4"
            video_path.write_text("fake video content")
            
            # Mock timeout
            def mock_extraction(*args, **kwargs):
                if "ffmpeg" in args[0][0] and len(args[0]) > 5:  # Extraction command
                    from subprocess import TimeoutExpired
                    raise TimeoutExpired(args[0], kwargs.get('timeout', 300))
                return Mock(returncode=0, stderr="")
            
            mock_run.side_effect = mock_extraction
            
            with pytest.raises(AudioExtractionError, match="timed out"):
                service.extract_audio(str(video_path))
    
    def test_get_audio_codec(self):
        """Test audio codec selection."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            assert service._get_audio_codec("wav") == "pcm_s16le"
            assert service._get_audio_codec("mp3") == "libmp3lame"
            assert service._get_audio_codec("flac") == "flac"
            assert service._get_audio_codec("unknown") == "pcm_s16le"  # Default
    
    def test_cleanup_audio_file(self):
        """Test audio file cleanup."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            # Create fake audio file
            audio_path = Path(self.temp_dir) / "test_audio.wav"
            audio_path.write_text("fake audio content")
            
            assert audio_path.exists()
            
            # Cleanup
            service.cleanup_audio_file(str(audio_path))
            
            assert not audio_path.exists()
    
    def test_cleanup_nonexistent_file(self):
        """Test cleanup of nonexistent file."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            # Should not raise exception
            service.cleanup_audio_file("/nonexistent/audio.wav")
    
    def test_get_audio_info_success(self):
        """Test getting audio information."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            # Mock FFmpeg verification
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            # Mock FFprobe output
            ffprobe_output = {
                "streams": [{
                    "duration": "120.5",
                    "sample_rate": "44100",
                    "channels": "2",
                    "codec_name": "aac"
                }]
            }
            
            def mock_ffprobe(*args, **kwargs):
                if "ffprobe" in args[0][0]:
                    import json
                    return Mock(returncode=0, stdout=json.dumps(ffprobe_output))
                return Mock(returncode=0, stderr="")
            
            mock_run.side_effect = mock_ffprobe
            
            info = service.get_audio_info("/fake/video.mp4")
            
            assert info["duration"] == 120.5
            assert info["sample_rate"] == 44100
            assert info["channels"] == 2
            assert info["codec"] == "aac"
    
    def test_get_audio_info_no_streams(self):
        """Test getting audio info with no audio streams."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            # Mock FFmpeg verification
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            # Mock FFprobe output with no streams
            ffprobe_output = {"streams": []}
            
            def mock_ffprobe(*args, **kwargs):
                if "ffprobe" in args[0][0]:
                    import json
                    return Mock(returncode=0, stdout=json.dumps(ffprobe_output))
                return Mock(returncode=0, stderr="")
            
            mock_run.side_effect = mock_ffprobe
            
            with pytest.raises(AudioExtractionError, match="No audio streams found"):
                service.get_audio_info("/fake/video.mp4")
    
    def test_get_audio_info_ffprobe_failure(self):
        """Test getting audio info with FFprobe failure."""
        with patch('src.services.audio_extraction_service.subprocess.run') as mock_run:
            # Mock FFmpeg verification
            mock_run.return_value.returncode = 0
            service = AudioExtractionService(self.temp_dir)
            
            def mock_ffprobe(*args, **kwargs):
                if "ffprobe" in args[0][0]:
                    return Mock(returncode=1, stderr="FFprobe error")
                return Mock(returncode=0, stderr="")
            
            mock_run.side_effect = mock_ffprobe
            
            with pytest.raises(AudioExtractionError, match="FFprobe failed"):
                service.get_audio_info("/fake/video.mp4")
