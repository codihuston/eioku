from sqlalchemy.orm import Session

from ..database.models import Place as PlaceEntity
from ..domain.models import Place
from .interfaces import PlaceRepository


class SqlPlaceRepository(PlaceRepository):
    """SQLAlchemy implementation of PlaceRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, place: Place) -> Place:
        """Save place to database."""
        entity = self._to_entity(place)

        # Check if exists (update) or new (create)
        existing = (
            self.session.query(PlaceEntity)
            .filter(PlaceEntity.place_id == place.place_id)
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

    def find_by_video_id(self, video_id: str) -> list[Place]:
        """Find all places for a video."""
        entities = (
            self.session.query(PlaceEntity)
            .filter(PlaceEntity.video_id == video_id)
            .all()
        )
        return [self._to_domain(entity) for entity in entities]

    def find_by_label(self, video_id: str, label: str) -> list[Place]:
        """Find places by label within a video."""
        entities = (
            self.session.query(PlaceEntity)
            .filter(PlaceEntity.video_id == video_id, PlaceEntity.label == label)
            .all()
        )
        return [self._to_domain(entity) for entity in entities]

    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all places for a video."""
        deleted_count = (
            self.session.query(PlaceEntity)
            .filter(PlaceEntity.video_id == video_id)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _to_entity(self, domain: Place) -> PlaceEntity:
        """Convert domain model to SQLAlchemy entity."""
        return PlaceEntity(
            place_id=domain.place_id,
            video_id=domain.video_id,
            label=domain.label,
            timestamps=domain.timestamps,
            confidence=domain.confidence,
            alternative_labels=domain.alternative_labels,
            metadata=domain.metadata,
        )

    def _to_domain(self, entity: PlaceEntity) -> Place:
        """Convert SQLAlchemy entity to domain model."""
        return Place(
            place_id=entity.place_id,
            video_id=entity.video_id,
            label=entity.label,
            timestamps=entity.timestamps,
            confidence=entity.confidence,
            alternative_labels=entity.alternative_labels,
            metadata=entity.metadata,
            created_at=entity.created_at,
        )
