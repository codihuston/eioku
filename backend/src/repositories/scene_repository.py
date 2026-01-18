from sqlalchemy.orm import Session

from ..database.models import Scene as SceneEntity
from ..domain.models import Scene
from .interfaces import SceneRepository


class SqlSceneRepository(SceneRepository):
    """SQLAlchemy implementation of SceneRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, scene: Scene) -> Scene:
        """Save scene to database."""
        entity = self._to_entity(scene)

        # Check if exists (update) or new (create)
        existing = (
            self.session.query(SceneEntity)
            .filter(SceneEntity.scene_id == scene.scene_id)
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

    def find_by_video_id(self, video_id: str) -> list[Scene]:
        """Find all scenes for a video."""
        entities = (
            self.session.query(SceneEntity)
            .filter(SceneEntity.video_id == video_id)
            .order_by(SceneEntity.scene)
            .all()
        )
        return [self._to_domain(entity) for entity in entities]

    def find_by_scene_number(self, video_id: str, scene_number: int) -> Scene | None:
        """Find scene by number within a video."""
        entity = (
            self.session.query(SceneEntity)
            .filter(SceneEntity.video_id == video_id, SceneEntity.scene == scene_number)
            .first()
        )
        return self._to_domain(entity) if entity else None

    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all scenes for a video."""
        deleted_count = (
            self.session.query(SceneEntity)
            .filter(SceneEntity.video_id == video_id)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _to_entity(self, domain: Scene) -> SceneEntity:
        """Convert domain model to SQLAlchemy entity."""
        return SceneEntity(
            scene_id=domain.scene_id,
            video_id=domain.video_id,
            scene=domain.scene,
            start=domain.start,
            end=domain.end,
            thumbnail_path=domain.thumbnail_path,
        )

    def _to_domain(self, entity: SceneEntity) -> Scene:
        """Convert SQLAlchemy entity to domain model."""
        return Scene(
            scene_id=entity.scene_id,
            video_id=entity.video_id,
            scene=entity.scene,
            start=entity.start,
            end=entity.end,
            thumbnail_path=entity.thumbnail_path,
            created_at=entity.created_at,
        )
