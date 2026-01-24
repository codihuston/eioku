"""Repository implementation for artifact envelope storage."""

import json
import logging

from sqlalchemy.orm import Session

from ..database.models import Artifact as ArtifactEntity
from ..domain.artifacts import ArtifactEnvelope, SelectionPolicy
from ..domain.schema_registry import SchemaRegistry
from ..services.projection_sync_service import ProjectionSyncService
from .interfaces import ArtifactRepository

logger = logging.getLogger(__name__)


class SqlArtifactRepository(ArtifactRepository):
    """SQLAlchemy implementation of ArtifactRepository."""

    def __init__(
        self,
        session: Session,
        schema_registry: SchemaRegistry,
        projection_sync_service: ProjectionSyncService | None = None,
    ):
        self.session = session
        self.schema_registry = schema_registry
        self.projection_sync_service = projection_sync_service or ProjectionSyncService(
            session
        )

    def create(self, artifact: ArtifactEnvelope) -> ArtifactEnvelope:
        """Create a new artifact with schema validation."""
        logger.debug(
            f"ArtifactRepository.create() called for artifact: {artifact.artifact_id}"
        )

        # Validate payload against schema
        payload_dict = json.loads(artifact.payload_json)
        self.schema_registry.validate(
            artifact.artifact_type, artifact.schema_version, payload_dict
        )
        logger.debug(f"Payload validated for artifact type: {artifact.artifact_type}")

        # Convert domain model to entity
        entity = self._to_entity(artifact)

        # Store in database
        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)

        logger.info(
            f"Created artifact: {artifact.artifact_id}, "
            f"type: {artifact.artifact_type}, run: {artifact.run_id}"
        )

        # Synchronize to projection tables
        try:
            self.projection_sync_service.sync_artifact(artifact)
            logger.debug(f"Synced artifact {artifact.artifact_id} to projections")
        except Exception as e:
            # Log error but don't fail the artifact creation
            # Projections can be rebuilt later if needed
            logger.error(
                f"Failed to sync projection for artifact {artifact.artifact_id}: {e}"
            )

        return self._to_domain(entity)

    def batch_create(self, artifacts: list[ArtifactEnvelope]) -> list[ArtifactEnvelope]:
        """Create multiple artifacts in a single transaction.

        This is more efficient than calling create() multiple times as it:
        1. Validates all artifacts first
        2. Inserts all in one database round-trip
        3. Syncs all projections together

        Args:
            artifacts: List of artifacts to create

        Returns:
            List of created artifacts

        Raises:
            ValidationError: If any artifact fails schema validation
            DatabaseError: If database operation fails
        """
        if not artifacts:
            return []

        logger.debug(
            f"ArtifactRepository.batch_create() called for {len(artifacts)} artifacts"
        )

        try:
            # Step 1: Validate all payloads first (fail fast)
            for artifact in artifacts:
                payload_dict = json.loads(artifact.payload_json)
                self.schema_registry.validate(
                    artifact.artifact_type, artifact.schema_version, payload_dict
                )
            logger.debug(f"All {len(artifacts)} payloads validated")

            # Step 2: Convert all to entities
            entities = [self._to_entity(artifact) for artifact in artifacts]

            # Step 3: Bulk insert in a single transaction
            self.session.bulk_save_objects(entities)
            self.session.commit()

            logger.info(
                f"Created {len(artifacts)} artifacts in batch, "
                f"types: {set(a.artifact_type for a in artifacts)}"
            )

            # Step 4: Synchronize to projection tables
            for artifact in artifacts:
                try:
                    self.projection_sync_service.sync_artifact(artifact)
                except Exception as e:
                    # Log error but don't fail the batch
                    artifact_id = artifact.artifact_id
                    logger.error(
                        f"Failed to sync projection for artifact {artifact_id}: {e}"
                    )

            return artifacts

        except Exception as e:
            # Rollback entire batch on any error
            logger.error(f"Batch create failed, rolling back: {e}")
            self.session.rollback()
            raise

    def get_by_id(self, artifact_id: str) -> ArtifactEnvelope | None:
        """Get artifact by ID."""
        entity = (
            self.session.query(ArtifactEntity)
            .filter(ArtifactEntity.artifact_id == artifact_id)
            .first()
        )
        return self._to_domain(entity) if entity else None

    def get_by_asset(
        self,
        asset_id: str,
        artifact_type: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        selection: SelectionPolicy | None = None,
    ) -> list[ArtifactEnvelope]:
        """Get artifacts for an asset with optional filtering."""
        query = self.session.query(ArtifactEntity).filter(
            ArtifactEntity.asset_id == asset_id
        )

        if artifact_type:
            query = query.filter(ArtifactEntity.artifact_type == artifact_type)

        if start_ms is not None:
            query = query.filter(ArtifactEntity.span_start_ms >= start_ms)

        if end_ms is not None:
            query = query.filter(ArtifactEntity.span_end_ms <= end_ms)

        # Apply selection policy
        if selection:
            query = self._apply_selection_policy(
                query, asset_id, artifact_type, selection
            )

        query = query.order_by(ArtifactEntity.span_start_ms)
        entities = query.all()
        return [self._to_domain(e) for e in entities]

    def get_by_span(
        self,
        asset_id: str,
        artifact_type: str,
        span_start_ms: int,
        span_end_ms: int,
        selection: SelectionPolicy | None = None,
    ) -> list[ArtifactEnvelope]:
        """Get artifacts overlapping a time span."""
        query = self.session.query(ArtifactEntity).filter(
            ArtifactEntity.asset_id == asset_id,
            ArtifactEntity.artifact_type == artifact_type,
            ArtifactEntity.span_start_ms < span_end_ms,
            ArtifactEntity.span_end_ms > span_start_ms,
        )

        if selection:
            query = self._apply_selection_policy(
                query, asset_id, artifact_type, selection
            )

        entities = query.all()
        return [self._to_domain(e) for e in entities]

    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        deleted = (
            self.session.query(ArtifactEntity)
            .filter(ArtifactEntity.artifact_id == artifact_id)
            .delete()
        )
        self.session.commit()
        return deleted > 0

    def _apply_selection_policy(self, query, asset_id, artifact_type, policy):
        """Apply selection policy to query."""
        if policy.mode == "pinned" and policy.pinned_run_id:
            query = query.filter(ArtifactEntity.run_id == policy.pinned_run_id)
            logger.debug(f"Applied pinned selection: run_id={policy.pinned_run_id}")
        elif policy.mode == "profile" and policy.preferred_profile:
            query = query.filter(
                ArtifactEntity.model_profile == policy.preferred_profile
            )
            logger.debug(
                f"Applied profile selection: profile={policy.preferred_profile}"
            )
        elif policy.mode == "latest":
            # Get most recent run_id for this asset/type
            subquery = (
                self.session.query(ArtifactEntity.run_id)
                .filter(
                    ArtifactEntity.asset_id == asset_id,
                    ArtifactEntity.artifact_type == artifact_type,
                )
                .order_by(ArtifactEntity.created_at.desc())
                .limit(1)
                .scalar_subquery()
            )
            query = query.filter(ArtifactEntity.run_id == subquery)
            logger.debug("Applied latest selection")
        elif policy.mode == "best_quality":
            # Prefer high_quality > balanced > fast
            query = query.order_by(
                self.session.query(ArtifactEntity)
                .filter(ArtifactEntity.model_profile == "high_quality")
                .exists()
                .desc(),
                self.session.query(ArtifactEntity)
                .filter(ArtifactEntity.model_profile == "balanced")
                .exists()
                .desc(),
            )
            logger.debug("Applied best_quality selection")

        return query

    def _to_entity(self, domain: ArtifactEnvelope) -> ArtifactEntity:
        """Convert domain model to SQLAlchemy entity."""
        return ArtifactEntity(
            artifact_id=domain.artifact_id,
            asset_id=domain.asset_id,
            artifact_type=domain.artifact_type,
            schema_version=domain.schema_version,
            span_start_ms=domain.span_start_ms,
            span_end_ms=domain.span_end_ms,
            payload_json=domain.payload_json,
            producer=domain.producer,
            producer_version=domain.producer_version,
            model_profile=domain.model_profile,
            config_hash=domain.config_hash,
            input_hash=domain.input_hash,
            run_id=domain.run_id,
            created_at=domain.created_at,
        )

    def _to_domain(self, entity: ArtifactEntity) -> ArtifactEnvelope:
        """Convert SQLAlchemy entity to domain model."""
        return ArtifactEnvelope(
            artifact_id=entity.artifact_id,
            asset_id=entity.asset_id,
            artifact_type=entity.artifact_type,
            schema_version=entity.schema_version,
            span_start_ms=entity.span_start_ms,
            span_end_ms=entity.span_end_ms,
            payload_json=json.dumps(entity.payload_json)
            if isinstance(entity.payload_json, dict)
            else entity.payload_json,
            producer=entity.producer,
            producer_version=entity.producer_version,
            model_profile=entity.model_profile,
            config_hash=entity.config_hash,
            input_hash=entity.input_hash,
            run_id=entity.run_id,
            created_at=entity.created_at,
        )
