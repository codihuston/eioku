from sqlalchemy.orm import Session

from ..database.models import Object as ObjectEntity
from ..domain.models import Object
from .interfaces import ObjectRepository


class SqlObjectRepository(ObjectRepository):
    """SQLAlchemy implementation of ObjectRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, obj: Object) -> Object:
        """Save object to database."""
        entity = self._to_entity(obj)

        # Check if exists (update) or new (create)
        existing = (
            self.session.query(ObjectEntity)
            .filter(ObjectEntity.object_id == obj.object_id)
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

    def find_by_video_id(self, video_id: str) -> list[Object]:
        """Find all objects for a video."""
        entities = (
            self.session.query(ObjectEntity)
            .filter(ObjectEntity.video_id == video_id)
            .all()
        )
        return [self._to_domain(entity) for entity in entities]

    def find_by_label(self, video_id: str, label: str) -> list[Object]:
        """Find objects by label within a video."""
        entities = (
            self.session.query(ObjectEntity)
            .filter(ObjectEntity.video_id == video_id, ObjectEntity.label == label)
            .all()
        )
        return [self._to_domain(entity) for entity in entities]

    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all objects for a video."""
        deleted_count = (
            self.session.query(ObjectEntity)
            .filter(ObjectEntity.video_id == video_id)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _to_entity(self, domain: Object) -> ObjectEntity:
        """Convert domain model to SQLAlchemy entity."""
        return ObjectEntity(
            object_id=domain.object_id,
            video_id=domain.video_id,
            label=domain.label,
            timestamps=domain.timestamps,
            bounding_boxes=domain.bounding_boxes,
        )

    def _to_domain(self, entity: ObjectEntity) -> Object:
        """Convert SQLAlchemy entity to domain model."""
        return Object(
            object_id=entity.object_id,
            video_id=entity.video_id,
            label=entity.label,
            timestamps=entity.timestamps,
            bounding_boxes=entity.bounding_boxes,
            created_at=entity.created_at,
        )
