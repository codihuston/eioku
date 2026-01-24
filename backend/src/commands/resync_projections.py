"""Management command to resync artifacts to projection tables."""

import logging

from src.database.connection import SessionLocal
from src.database.models import Artifact as ArtifactEntity
from src.domain.artifacts import ArtifactEnvelope
from src.domain.schema_initialization import register_all_schemas
from src.services.projection_sync_service import ProjectionSyncService

logger = logging.getLogger(__name__)


def resync_all_projections():
    """Resync all artifacts to their projection tables."""
    # Get all artifacts
    session = SessionLocal()
    artifacts = session.query(ArtifactEntity).all()
    session.close()

    logger.info(f"Found {len(artifacts)} artifacts to resync")

    synced_count = 0
    failed_count = 0

    for entity in artifacts:
        # Create a new session for each artifact to avoid transaction issues
        session = SessionLocal()
        try:
            projection_sync = ProjectionSyncService(session)

            # Convert entity to domain model
            artifact = ArtifactEnvelope(
                artifact_id=entity.artifact_id,
                asset_id=entity.asset_id,
                artifact_type=entity.artifact_type,
                schema_version=entity.schema_version,
                span_start_ms=entity.span_start_ms,
                span_end_ms=entity.span_end_ms,
                payload_json=entity.payload_json,
                producer=entity.producer,
                producer_version=entity.producer_version,
                model_profile=entity.model_profile,
                config_hash=entity.config_hash,
                input_hash=entity.input_hash,
                run_id=entity.run_id,
                created_at=entity.created_at,
            )

            # Sync to projection
            projection_sync.sync_artifact(artifact)
            session.commit()
            synced_count += 1

            if synced_count % 10 == 0:
                logger.info(f"Synced {synced_count} artifacts...")

        except Exception as e:
            session.rollback()
            failed_count += 1
            logger.error(f"Failed to sync artifact {entity.artifact_id}: {e}")

        finally:
            session.close()

    logger.info(f"Resync complete: {synced_count} synced, {failed_count} failed")
    return synced_count, failed_count


if __name__ == "__main__":
    # Register schemas first
    register_all_schemas()
    resync_all_projections()
