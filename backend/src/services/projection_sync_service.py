"""Service for synchronizing artifact projections."""

import json
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..domain.artifacts import ArtifactEnvelope

logger = logging.getLogger(__name__)


class ProjectionSyncError(Exception):
    """Raised when projection synchronization fails."""

    pass


class ProjectionSyncService:
    """Service for synchronizing artifact data to projection tables."""

    def __init__(self, session: Session):
        self.session = session

    def sync_artifact(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize an artifact to its projection tables.

        Args:
            artifact: The artifact to synchronize

        Raises:
            ProjectionSyncError: If synchronization fails
        """
        try:
            if artifact.artifact_type == "transcript.segment":
                self._sync_transcript_fts(artifact)
            elif artifact.artifact_type == "scene":
                self._sync_scene_ranges(artifact)
            # Add more artifact types here as they are implemented
            # elif artifact.artifact_type == "object.detection":
            #     self._sync_object_labels(artifact)
            # etc.

        except Exception as e:
            error_msg = (
                f"Failed to sync projection for artifact {artifact.artifact_id}: {e}"
            )
            logger.error(error_msg)
            raise ProjectionSyncError(error_msg) from e

    def _sync_transcript_fts(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize transcript artifact to FTS projection.

        Args:
            artifact: The transcript.segment artifact to synchronize
        """
        # Parse payload to extract text
        payload = json.loads(artifact.payload_json)
        transcript_text = payload.get("text", "")

        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        if is_postgresql:
            # PostgreSQL: Insert into transcript_fts table
            # The tsvector column is automatically computed
            sql = text(
                """
                INSERT INTO transcript_fts
                    (artifact_id, asset_id, start_ms, end_ms, text)
                VALUES (:artifact_id, :asset_id, :start_ms, :end_ms, :text)
                ON CONFLICT (artifact_id) DO UPDATE
                SET asset_id = EXCLUDED.asset_id,
                    start_ms = EXCLUDED.start_ms,
                    end_ms = EXCLUDED.end_ms,
                    text = EXCLUDED.text
                """
            )
        else:
            # SQLite: Insert into FTS5 virtual table and metadata table
            # First, insert into metadata table
            metadata_sql = text(
                """
                INSERT OR REPLACE INTO transcript_fts_metadata
                    (artifact_id, asset_id, start_ms, end_ms)
                VALUES (:artifact_id, :asset_id, :start_ms, :end_ms)
                """
            )

            self.session.execute(
                metadata_sql,
                {
                    "artifact_id": artifact.artifact_id,
                    "asset_id": artifact.asset_id,
                    "start_ms": artifact.span_start_ms,
                    "end_ms": artifact.span_end_ms,
                },
            )

            # Then, insert into FTS5 table
            sql = text(
                """
                INSERT INTO transcript_fts
                    (artifact_id, asset_id, start_ms, end_ms, text)
                VALUES (:artifact_id, :asset_id, :start_ms, :end_ms, :text)
                """
            )

        self.session.execute(
            sql,
            {
                "artifact_id": artifact.artifact_id,
                "asset_id": artifact.asset_id,
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms,
                "text": transcript_text,
            },
        )

        self.session.commit()

        logger.debug(
            f"Synced transcript artifact {artifact.artifact_id} to FTS projection"
        )

    def _sync_scene_ranges(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize scene artifact to scene_ranges projection.

        Args:
            artifact: The scene artifact to synchronize
        """
        # Parse payload to extract scene_index
        payload = json.loads(artifact.payload_json)
        scene_index = payload.get("scene_index", 0)

        # Insert into scene_ranges projection table
        sql = text(
            """
            INSERT OR REPLACE INTO scene_ranges
                (artifact_id, asset_id, scene_index, start_ms, end_ms)
            VALUES (:artifact_id, :asset_id, :scene_index, :start_ms, :end_ms)
            """
        )

        self.session.execute(
            sql,
            {
                "artifact_id": artifact.artifact_id,
                "asset_id": artifact.asset_id,
                "scene_index": scene_index,
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms,
            },
        )

        self.session.commit()

        logger.debug(
            f"Synced scene artifact {artifact.artifact_id} to scene_ranges projection"
        )
