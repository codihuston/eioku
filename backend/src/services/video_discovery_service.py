"""Video file discovery service."""

from pathlib import Path

from ..domain.models import PathConfig, Video
from ..repositories.interfaces import VideoRepository
from ..utils.print_logger import get_logger
from .path_config_manager import PathConfigManager

# Configure logging
logger = get_logger(__name__)


class VideoDiscoveryService:
    """Service for discovering video files in configured paths."""

    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}

    def __init__(
        self,
        path_config_manager: PathConfigManager,
        video_repository: VideoRepository,
    ):
        self.path_config_manager = path_config_manager
        self.video_repository = video_repository

    def discover_videos(self) -> list[Video]:
        """Discover all video files in configured paths."""
        logger.info("Starting discovery...")
        discovered_videos = []
        path_configs = self.path_config_manager.list_paths()
        logger.info(f"Found {len(path_configs)} configured paths")

        for path_config in path_configs:
            logger.info(f"Scanning path: {path_config.path}")
            videos = self._scan_path(path_config)
            logger.info(f"Found {len(videos)} videos in {path_config.path}")
            discovered_videos.extend(videos)

        logger.info(f"Total discovered videos: {len(discovered_videos)}")
        return discovered_videos

    def _scan_path(self, path_config: PathConfig) -> list[Video]:
        """Scan a single path configuration for video files."""
        videos = []
        path = Path(path_config.path)

        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return videos

        # Use glob patterns for each supported format (much more efficient)
        for extension in self.SUPPORTED_FORMATS:
            pattern = f"**/*{extension}" if path_config.recursive else f"*{extension}"
            logger.debug(f"Scanning with pattern: {pattern}")

            glob_method = path.rglob if path_config.recursive else path.glob
            for video_file in glob_method(f"*{extension}"):
                if video_file.is_file():
                    video = self._create_video_from_file(video_file)
                    if video:
                        videos.append(video)

        return videos

    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS

    def _create_video_from_file(self, file_path: Path) -> Video | None:
        """Create Video domain object from file path."""
        try:
            logger.debug(f"Creating video from file: {file_path}")

            # Check if video already exists in database
            existing = self.video_repository.find_by_path(str(file_path))
            if existing:
                logger.debug(f"Video already exists: {existing.video_id}")
                return existing

            # Get file stats
            stat = file_path.stat()
            logger.debug(
                f"File stats - size: {stat.st_size}, modified: {stat.st_mtime}"
            )

            # Create new video
            import uuid
            from datetime import datetime

            video = Video(
                video_id=str(uuid.uuid4()),
                file_path=str(file_path),
                filename=file_path.name,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                status="discovered",
                file_size=stat.st_size,
            )
            logger.debug(f"Created video object: {video.video_id}")

            # Save to database
            logger.debug("Attempting to save video to database...")
            saved_video = self.video_repository.save(video)
            logger.info(f"Video saved successfully: {saved_video.video_id}")
            return saved_video

        except Exception as e:
            # Log the error for debugging
            logger.error(f"Error creating video from {file_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def validate_existing_videos(self) -> list[Video]:
        """Validate existing videos exist on filesystem and clean up orphaned tasks."""
        missing_videos = []
        # Check all videos that have been discovered or completed
        all_videos = []
        all_videos.extend(self.video_repository.find_by_status("discovered"))
        all_videos.extend(self.video_repository.find_by_status("completed"))
        all_videos.extend(self.video_repository.find_by_status("hashed"))
        all_videos.extend(self.video_repository.find_by_status("processing"))

        for video in all_videos:
            if not Path(video.file_path).exists():
                logger.info(
                    f"Video missing from filesystem: {video.filename} "
                    f"at {video.file_path}"
                )

                # Delete the video from database
                self.video_repository.delete(video.video_id)
                missing_videos.append(video)

                logger.info(f"Removed missing video {video.video_id} from database")

        return missing_videos
