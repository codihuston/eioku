import logging

from sqlalchemy.orm import Session

from ..database.models import Video as VideoEntity
from ..domain.models import Video
from .interfaces import VideoRepository

logger = logging.getLogger(__name__)


class SqlVideoRepository(VideoRepository):
    """SQLAlchemy implementation of VideoRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, video: Video) -> Video:
        """Save video to database."""
        logger.debug(f"VideoRepository.save() called for video: {video.video_id}")
        try:
            # Convert domain model to entity
            entity = self._to_entity(video)
            logger.debug(f"Converted to entity: {entity.video_id}")

            # Check if exists (update) or new (create)
            existing = (
                self.session.query(VideoEntity)
                .filter(VideoEntity.video_id == video.video_id)
                .first()
            )

            if existing:
                logger.debug(f"Updating existing video: {existing.video_id}")
                # Update existing
                for key, value in entity.__dict__.items():
                    if not key.startswith("_") and value is not None:
                        setattr(existing, key, value)
                self.session.commit()
                logger.debug("Update committed successfully")
                self.session.refresh(existing)
                return self._to_domain(existing)
            else:
                logger.debug(f"Creating new video: {entity.video_id}")
                # Create new
                self.session.add(entity)
                self.session.commit()
                logger.debug("New video committed successfully")
                self.session.refresh(entity)
                result = self._to_domain(entity)
                logger.debug(f"Returning saved video: {result.video_id}")
                return result
        except Exception as e:
            logger.error(f"Error in VideoRepository.save(): {e}")
            import traceback

            traceback.print_exc()
            raise

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

    def find_all(self) -> list[Video]:
        """Find all videos."""
        entities = self.session.query(VideoEntity).all()
        return [self._to_domain(entity) for entity in entities]

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
