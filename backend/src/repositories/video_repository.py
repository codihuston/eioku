from sqlalchemy.orm import Session

from ..database.models import Video as VideoEntity
from ..domain.models import Video
from .interfaces import VideoRepository


class SqlVideoRepository(VideoRepository):
    """SQLAlchemy implementation of VideoRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, video: Video) -> Video:
        """Save video to database."""
        # Convert domain model to entity
        entity = self._to_entity(video)

        # Check if exists (update) or new (create)
        existing = (
            self.session.query(VideoEntity)
            .filter(VideoEntity.video_id == video.video_id)
            .first()
        )

        if existing:
            # Update existing
            for key, value in entity.__dict__.items():
                if not key.startswith("_") and value is not None:
                    setattr(existing, key, value)
            self.session.commit()
            self.session.refresh(existing)
            return self._to_domain(existing)
        else:
            # Create new
            self.session.add(entity)
            self.session.commit()
            self.session.refresh(entity)
            return self._to_domain(entity)

    def find_by_id(self, video_id: str) -> Video | None:
        """Find video by ID."""
        entity = (
            self.session.query(VideoEntity)
            .filter(VideoEntity.video_id == video_id)
            .first()
        )
        return self._to_domain(entity) if entity else None

    def find_by_path(self, file_path: str) -> Video | None:
        """Find video by file path."""
        entity = (
            self.session.query(VideoEntity)
            .filter(VideoEntity.file_path == file_path)
            .first()
        )
        return self._to_domain(entity) if entity else None

    def find_by_status(self, status: str) -> list[Video]:
        """Find videos by status."""
        entities = (
            self.session.query(VideoEntity).filter(VideoEntity.status == status).all()
        )
        return [self._to_domain(entity) for entity in entities]

    def delete(self, video_id: str) -> bool:
        """Delete video by ID."""
        entity = (
            self.session.query(VideoEntity)
            .filter(VideoEntity.video_id == video_id)
            .first()
        )
        if entity:
            self.session.delete(entity)
            self.session.commit()
            return True
        return False

    def _to_entity(self, domain: Video) -> VideoEntity:
        """Convert domain model to SQLAlchemy entity."""
        return VideoEntity(
            video_id=domain.video_id,
            file_path=domain.file_path,
            filename=domain.filename,
            file_hash=domain.file_hash,
            last_modified=domain.last_modified,
            status=domain.status,
            duration=domain.duration,
            file_size=domain.file_size,
            processed_at=domain.processed_at,
        )

    def _to_domain(self, entity: VideoEntity) -> Video:
        """Convert SQLAlchemy entity to domain model."""
        return Video(
            video_id=entity.video_id,
            file_path=entity.file_path,
            filename=entity.filename,
            file_hash=entity.file_hash,
            last_modified=entity.last_modified,
            status=entity.status,
            duration=entity.duration,
            file_size=entity.file_size,
            processed_at=entity.processed_at,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )
