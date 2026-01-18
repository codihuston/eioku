"""Video file discovery service."""

from pathlib import Path

from ..domain.models import PathConfig, Video
from ..repositories.interfaces import VideoRepository
from .path_config_manager import PathConfigManager


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
        discovered_videos = []
        path_configs = self.path_config_manager.list_paths()

        for path_config in path_configs:
            videos = self._scan_path(path_config)
            discovered_videos.extend(videos)

        return discovered_videos

    def _scan_path(self, path_config: PathConfig) -> list[Video]:
        """Scan a single path configuration for video files."""
        videos = []
        path = Path(path_config.path)

        if not path.exists():
            return videos

        if path_config.recursive:
            # Recursive scan
            for video_file in path.rglob("*"):
                if self._is_video_file(video_file):
                    video = self._create_video_from_file(video_file)
                    if video:
                        videos.append(video)
        else:
            # Non-recursive scan
            for video_file in path.iterdir():
                if video_file.is_file() and self._is_video_file(video_file):
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
            # Check if video already exists in database
            existing = self.video_repository.find_by_path(str(file_path))
            if existing:
                return existing

            # Get file stats
            stat = file_path.stat()

            # Create new video
            import uuid
            from datetime import datetime

            video = Video(
                video_id=str(uuid.uuid4()),
                file_path=str(file_path),
                filename=file_path.name,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                file_size=stat.st_size,
                status="discovered",
            )

            return self.video_repository.save(video)

        except (OSError, PermissionError):
            # Skip files we can't access
            return None

    def validate_existing_videos(self) -> list[Video]:
        """Validate that existing videos still exist on filesystem."""
        missing_videos = []
        all_videos = self.video_repository.find_by_status("completed")

        for video in all_videos:
            if not Path(video.file_path).exists():
                video.status = "missing"
                self.video_repository.save(video)
                missing_videos.append(video)

        return missing_videos
