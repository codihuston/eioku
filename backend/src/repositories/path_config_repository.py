"""SQLAlchemy implementation of PathConfigRepository."""

from datetime import datetime

from sqlalchemy.orm import Session

from ..database.models import PathConfig as PathConfigEntity
from ..domain.models import PathConfig
from .interfaces import PathConfigRepository


class SQLAlchemyPathConfigRepository(PathConfigRepository):
    """SQLAlchemy implementation of PathConfigRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, path_config: PathConfig) -> PathConfig:
        """Save path config to database."""
        entity = PathConfigEntity(
            path_id=path_config.path_id,
            path=path_config.path,
            recursive="true" if path_config.recursive else "false",
            added_at=path_config.added_at or datetime.utcnow(),
        )

        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)

        return self._entity_to_domain(entity)

    def find_all(self) -> list[PathConfig]:
        """Find all configured paths."""
        entities = (
            self.session.query(PathConfigEntity)
            .order_by(PathConfigEntity.added_at.desc())
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def find_by_path(self, path: str) -> PathConfig | None:
        """Find path config by path."""
        entity = (
            self.session.query(PathConfigEntity)
            .filter(PathConfigEntity.path == path)
            .first()
        )
        return self._entity_to_domain(entity) if entity else None

    def delete_by_path(self, path: str) -> bool:
        """Delete path config by path."""
        deleted_count = (
            self.session.query(PathConfigEntity)
            .filter(PathConfigEntity.path == path)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _entity_to_domain(self, entity: PathConfigEntity) -> PathConfig:
        """Convert database entity to domain model."""
        return PathConfig(
            path_id=entity.path_id,
            path=entity.path,
            recursive=entity.recursive == "true",
            added_at=entity.added_at,
        )
