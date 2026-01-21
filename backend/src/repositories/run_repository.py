"""Repository implementation for Run tracking."""

import logging

from sqlalchemy.orm import Session

from ..database.models import Run as RunEntity
from ..domain.artifacts import Run
from .interfaces import RunRepository

logger = logging.getLogger(__name__)


class SqlRunRepository(RunRepository):
    """SQLAlchemy implementation of RunRepository."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, run: Run) -> Run:
        """Create a new run record."""
        logger.debug(f"RunRepository.create() called for run: {run.run_id}")

        # Convert domain model to entity
        entity = self._to_entity(run)

        # Store in database
        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)

        logger.info(
            f"Created run: {run.run_id}, "
            f"asset: {run.asset_id}, "
            f"status: {run.status}"
        )

        return self._to_domain(entity)

    def get_by_id(self, run_id: str) -> Run | None:
        """Get run by ID."""
        entity = (
            self.session.query(RunEntity).filter(RunEntity.run_id == run_id).first()
        )
        return self._to_domain(entity) if entity else None

    def get_by_asset(self, asset_id: str) -> list[Run]:
        """Get all runs for an asset."""
        entities = (
            self.session.query(RunEntity)
            .filter(RunEntity.asset_id == asset_id)
            .order_by(RunEntity.started_at.desc())
            .all()
        )
        return [self._to_domain(e) for e in entities]

    def get_by_status(self, status: str) -> list[Run]:
        """Get all runs with a specific status."""
        entities = (
            self.session.query(RunEntity)
            .filter(RunEntity.status == status)
            .order_by(RunEntity.started_at.desc())
            .all()
        )
        return [self._to_domain(e) for e in entities]

    def update(self, run: Run) -> Run:
        """Update an existing run record."""
        logger.debug(f"RunRepository.update() called for run: {run.run_id}")

        entity = (
            self.session.query(RunEntity).filter(RunEntity.run_id == run.run_id).first()
        )

        if not entity:
            raise ValueError(f"Run not found: {run.run_id}")

        # Update entity fields
        entity.status = run.status
        entity.finished_at = run.finished_at
        entity.error = run.error

        self.session.commit()
        self.session.refresh(entity)

        logger.info(f"Updated run: {run.run_id}, status: {run.status}")

        return self._to_domain(entity)

    def delete(self, run_id: str) -> bool:
        """Delete a run record."""
        deleted = (
            self.session.query(RunEntity).filter(RunEntity.run_id == run_id).delete()
        )
        self.session.commit()
        return deleted > 0

    def _to_entity(self, domain: Run) -> RunEntity:
        """Convert domain model to SQLAlchemy entity."""
        return RunEntity(
            run_id=domain.run_id,
            asset_id=domain.asset_id,
            pipeline_profile=domain.pipeline_profile,
            started_at=domain.started_at,
            finished_at=domain.finished_at,
            status=domain.status,
            error=domain.error,
        )

    def _to_domain(self, entity: RunEntity) -> Run:
        """Convert SQLAlchemy entity to domain model."""
        return Run(
            run_id=entity.run_id,
            asset_id=entity.asset_id,
            pipeline_profile=entity.pipeline_profile,
            started_at=entity.started_at,
            status=entity.status,
            finished_at=entity.finished_at,
            error=entity.error,
        )
