from sqlalchemy.orm import Session

from .models import Video


class VideoDAO:
    """Data Access Object for Video operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, video: Video) -> Video:
        """Create a new video record."""
        self.session.add(video)
        self.session.commit()
        self.session.refresh(video)
        return video

    def get_by_id(self, video_id: str) -> Video | None:
        """Get video by ID."""
        return self.session.query(Video).filter(Video.video_id == video_id).first()

    def get_by_path(self, file_path: str) -> Video | None:
        """Get video by file path."""
        return self.session.query(Video).filter(Video.file_path == file_path).first()

    def get_by_status(self, status: str) -> list[Video]:
        """Get videos by processing status."""
        return self.session.query(Video).filter(Video.status == status).all()

    def update(self, video: Video) -> Video:
        """Update video record."""
        self.session.commit()
        self.session.refresh(video)
        return video

    def delete(self, video_id: str) -> bool:
        """Delete video by ID."""
        video = self.get_by_id(video_id)
        if video:
            self.session.delete(video)
            self.session.commit()
            return True
        return False
