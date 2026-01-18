from ..domain.models import Video
from ..repositories.interfaces import VideoRepository


class VideoService:
    """Service layer for Video business operations."""

    def __init__(self, video_repository: VideoRepository):
        self.video_repository = video_repository

    def create_video(self, video: Video) -> Video:
        """Create a new video for processing."""
        # Business logic: Check if video already exists
        existing = self.video_repository.find_by_path(video.file_path)
        if existing:
            raise ValueError(f"Video already exists at path: {video.file_path}")

        return self.video_repository.save(video)

    def get_video(self, video_id: str) -> Video | None:
        """Get video by ID."""
        return self.video_repository.find_by_id(video_id)

    def get_videos_by_status(self, status: str) -> list[Video]:
        """Get videos by processing status."""
        return self.video_repository.find_by_status(status)

    def update_video_status(self, video_id: str, status: str) -> Video | None:
        """Update video processing status."""
        video = self.video_repository.find_by_id(video_id)
        if not video:
            return None

        # Business logic: Status transitions
        if status == "processing":
            video.mark_as_processing()
        elif status == "completed":
            from datetime import datetime

            video.mark_as_completed(datetime.now())
        elif status == "failed":
            video.mark_as_failed()

        return self.video_repository.save(video)

    def delete_video(self, video_id: str) -> bool:
        """Delete video and all associated data."""
        # Business logic: Could add validation here
        return self.video_repository.delete(video_id)

    def get_pending_videos(self) -> list[Video]:
        """Get videos pending processing - used by orchestrator."""
        return self.video_repository.find_by_status("pending")

    def get_all_videos(self) -> list[Video]:
        """Get all videos regardless of status."""
        return self.video_repository.find_all()
