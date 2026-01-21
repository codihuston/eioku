"""Scene detection service using FFmpeg."""

import hashlib
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from ..domain.artifacts import ArtifactEnvelope
from ..domain.schemas.scene_v1 import SceneV1
from ..utils.print_logger import get_logger

logger = get_logger(__name__)


class SceneDetectionError(Exception):
    """Exception raised when scene detection fails."""

    pass


class SceneDetectionService:
    """Service for detecting scenes in videos using FFmpeg."""

    def __init__(
        self,
        threshold: float = 0.4,
        min_scene_len: float = 0.6,
        producer_version: str = "1.0.0",
    ):
        """Initialize scene detection service.

        Args:
            threshold: Detection threshold (0.0-1.0, lower = more sensitive)
            min_scene_len: Minimum scene length in seconds
            producer_version: Version of the scene detection producer
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.producer_version = producer_version

    def detect_scenes(
        self,
        video_path: str,
        video_id: str,
        run_id: str,
        model_profile: str = "balanced",
    ) -> list[ArtifactEnvelope]:
        """Detect scenes in a video file using FFmpeg.

        Args:
            video_path: Path to the video file
            video_id: ID of the video being processed
            run_id: ID of the pipeline run
            model_profile: Model profile (fast, balanced, high_quality)

        Returns:
            List of ArtifactEnvelope objects containing scene data

        Raises:
            SceneDetectionError: If scene detection fails
        """
        if not Path(video_path).exists():
            raise SceneDetectionError(f"Video file not found: {video_path}")

        logger.info(f"Detecting scenes in video: {video_path}")

        try:
            # Get video duration first
            duration = self._get_video_duration(video_path)
            logger.info(f"Video duration: {duration:.2f}s")

            # Detect scene changes using FFmpeg
            scene_timestamps = self._detect_scene_changes(video_path)
            logger.info(f"Found {len(scene_timestamps)} scene changes")

            # Convert timestamps to artifact envelopes
            artifacts = self._create_artifacts_from_timestamps(
                scene_timestamps, duration, video_id, video_path, run_id, model_profile
            )

            # Filter scenes by minimum length
            artifacts = [
                a
                for a in artifacts
                if (a.span_end_ms - a.span_start_ms) >= (self.min_scene_len * 1000)
            ]

            logger.info(
                f"Detected {len(artifacts)} scenes in video {video_id} "
                f"(duration: {duration:.2f}s)"
            )

            return artifacts

        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg failed for {video_path}: {e.stderr}"
            logger.error(error_msg)
            raise SceneDetectionError(error_msg)
        except Exception as e:
            error_msg = f"Scene detection failed for {video_path}: {e}"
            logger.error(error_msg)
            raise SceneDetectionError(error_msg)

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        return float(result.stdout.strip())

    def _detect_scene_changes(self, video_path: str) -> list[float]:
        """Detect scene changes using FFmpeg's scene detection filter.

        Args:
            video_path: Path to video file

        Returns:
            List of timestamps where scenes change
        """
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-filter:v",
            f"select='gt(scene\\,{self.threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Parse timestamps from stderr (ffmpeg outputs to stderr)
        timestamps = []
        # Look for lines like: "pts_time:1.234"
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                match = re.search(r"pts_time:(\d+\.?\d*)", line)
                if match:
                    timestamp = float(match.group(1))
                    timestamps.append(timestamp)

        return sorted(timestamps)

    def _create_artifacts_from_timestamps(
        self,
        timestamps: list[float],
        duration: float,
        video_id: str,
        video_path: str,
        run_id: str,
        model_profile: str,
    ) -> list[ArtifactEnvelope]:
        """Create ArtifactEnvelope objects from scene change timestamps.

        Args:
            timestamps: List of scene change timestamps
            duration: Total video duration
            video_id: Video ID
            video_path: Path to video file (for input hash)
            run_id: Pipeline run ID
            model_profile: Model profile

        Returns:
            List of ArtifactEnvelope objects
        """
        artifacts = []

        # Calculate config hash based on detection parameters
        config_str = f"threshold={self.threshold},min_scene_len={self.min_scene_len}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # Calculate input hash based on video path
        input_hash = hashlib.sha256(video_path.encode()).hexdigest()[:16]

        # If no scene changes detected, treat entire video as one scene
        if not timestamps:
            scene_payload = SceneV1(
                scene_index=0,
                method="content",
                score=0.0,
                frame_number=0,
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=video_id,
                artifact_type="scene",
                schema_version=1,
                span_start_ms=0,
                span_end_ms=int(duration * 1000),
                payload_json=scene_payload.model_dump_json(),
                producer="ffmpeg",
                producer_version=self.producer_version,
                model_profile=model_profile,
                config_hash=config_hash,
                input_hash=input_hash,
                run_id=run_id,
                created_at=datetime.utcnow(),
            )
            artifacts.append(artifact)
            return artifacts

        # Create artifacts from timestamps
        # First scene: 0 to first timestamp
        scene_payload = SceneV1(
            scene_index=0,
            method="content",
            score=1.0,  # First scene change has high score
            frame_number=0,
        )

        artifact = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id=video_id,
            artifact_type="scene",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=int(timestamps[0] * 1000),
            payload_json=scene_payload.model_dump_json(),
            producer="ffmpeg",
            producer_version=self.producer_version,
            model_profile=model_profile,
            config_hash=config_hash,
            input_hash=input_hash,
            run_id=run_id,
            created_at=datetime.utcnow(),
        )
        artifacts.append(artifact)

        # Middle scenes: between timestamps
        for i in range(len(timestamps) - 1):
            scene_payload = SceneV1(
                scene_index=i + 1,
                method="content",
                score=1.0,
                frame_number=int(
                    timestamps[i] * 30
                ),  # Approximate frame number at 30fps
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=video_id,
                artifact_type="scene",
                schema_version=1,
                span_start_ms=int(timestamps[i] * 1000),
                span_end_ms=int(timestamps[i + 1] * 1000),
                payload_json=scene_payload.model_dump_json(),
                producer="ffmpeg",
                producer_version=self.producer_version,
                model_profile=model_profile,
                config_hash=config_hash,
                input_hash=input_hash,
                run_id=run_id,
                created_at=datetime.utcnow(),
            )
            artifacts.append(artifact)

        # Last scene: last timestamp to end
        scene_payload = SceneV1(
            scene_index=len(timestamps),
            method="content",
            score=1.0,
            frame_number=int(timestamps[-1] * 30),  # Approximate frame number at 30fps
        )

        artifact = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id=video_id,
            artifact_type="scene",
            schema_version=1,
            span_start_ms=int(timestamps[-1] * 1000),
            span_end_ms=int(duration * 1000),
            payload_json=scene_payload.model_dump_json(),
            producer="ffmpeg",
            producer_version=self.producer_version,
            model_profile=model_profile,
            config_hash=config_hash,
            input_hash=input_hash,
            run_id=run_id,
            created_at=datetime.utcnow(),
        )
        artifacts.append(artifact)

        return artifacts

    def get_scene_info(self, artifacts: list[ArtifactEnvelope]) -> dict:
        """Get summary information about detected scenes.

        Args:
            artifacts: List of scene ArtifactEnvelope objects

        Returns:
            Dictionary with scene statistics
        """
        if not artifacts:
            return {
                "scene_count": 0,
                "total_duration": 0.0,
                "avg_scene_length": 0.0,
                "min_scene_length": 0.0,
                "max_scene_length": 0.0,
            }

        durations = [
            (a.span_end_ms - a.span_start_ms) / 1000.0 for a in artifacts
        ]  # Convert to seconds

        return {
            "scene_count": len(artifacts),
            "total_duration": artifacts[-1].span_end_ms / 1000.0
            if artifacts
            else 0.0,  # Convert to seconds
            "avg_scene_length": sum(durations) / len(durations),
            "min_scene_length": min(durations),
            "max_scene_length": max(durations),
        }
