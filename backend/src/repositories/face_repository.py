"""SQLAlchemy implementation of FaceRepository."""

from datetime import datetime

from sqlalchemy.orm import Session

from ..database.models import Face as FaceEntity
from ..domain.models import Face
from .interfaces import FaceRepository


class SQLAlchemyFaceRepository(FaceRepository):
    """SQLAlchemy implementation of FaceRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, face: Face) -> Face:
        """Save face to database."""
        entity = FaceEntity(
            face_id=face.face_id,
            video_id=face.video_id,
            person_id=face.person_id,
            timestamps=face.timestamps,
            bounding_boxes=face.bounding_boxes,
            confidence=face.confidence,
            created_at=face.created_at or datetime.utcnow(),
        )

        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)

        return self._entity_to_domain(entity)

    def find_by_video_id(self, video_id: str) -> list[Face]:
        """Find all faces for a video."""
        entities = (
            self.session.query(FaceEntity).filter(FaceEntity.video_id == video_id).all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def find_by_person_id(self, video_id: str, person_id: str) -> list[Face]:
        """Find faces by person ID within a video."""
        entities = (
            self.session.query(FaceEntity)
            .filter(FaceEntity.video_id == video_id, FaceEntity.person_id == person_id)
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all faces for a video."""
        deleted_count = (
            self.session.query(FaceEntity)
            .filter(FaceEntity.video_id == video_id)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _entity_to_domain(self, entity: FaceEntity) -> Face:
        """Convert database entity to domain model."""
        return Face(
            face_id=entity.face_id,
            video_id=entity.video_id,
            person_id=entity.person_id,
            timestamps=entity.timestamps,
            bounding_boxes=entity.bounding_boxes,
            confidence=entity.confidence,
            created_at=entity.created_at,
        )
