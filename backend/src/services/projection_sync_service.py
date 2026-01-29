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
            elif artifact.artifact_type == "object.detection":
                self._sync_object_labels(artifact)
            elif artifact.artifact_type == "face.detection":
                self._sync_face_clusters(artifact)
            elif artifact.artifact_type == "ocr.text":
                self._sync_ocr_fts(artifact)
            elif artifact.artifact_type == "video.metadata":
                self._sync_video_metadata(artifact)
            # Add more artifact types here as they are implemented
            # elif artifact.artifact_type == "place.classification":
            #     self._sync_place_labels(artifact)

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

        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        if is_postgresql:
            # PostgreSQL syntax
            sql = text(
                """
                INSERT INTO scene_ranges
                    (artifact_id, asset_id, scene_index, start_ms, end_ms)
                VALUES (:artifact_id, :asset_id, :scene_index, :start_ms, :end_ms)
                ON CONFLICT (artifact_id) DO UPDATE
                SET asset_id = EXCLUDED.asset_id,
                    scene_index = EXCLUDED.scene_index,
                    start_ms = EXCLUDED.start_ms,
                    end_ms = EXCLUDED.end_ms
                """
            )
        else:
            # SQLite syntax
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

        logger.debug(
            f"Synced scene artifact {artifact.artifact_id} to scene_ranges projection"
        )

    def _sync_object_labels(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize object.detection artifact to object_labels projection.

        Args:
            artifact: The object.detection artifact to synchronize
        """
        # Parse payload to extract label and confidence
        payload = json.loads(artifact.payload_json)
        label = payload.get("label", "")
        confidence = payload.get("confidence", 0.0)

        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        if is_postgresql:
            # PostgreSQL syntax
            sql = text(
                """
                INSERT INTO object_labels
                    (artifact_id, asset_id, label, confidence, start_ms, end_ms)
                VALUES (
                    :artifact_id, :asset_id, :label, :confidence,
                    :start_ms, :end_ms
                )
                ON CONFLICT (artifact_id) DO UPDATE
                SET asset_id = EXCLUDED.asset_id,
                    label = EXCLUDED.label,
                    confidence = EXCLUDED.confidence,
                    start_ms = EXCLUDED.start_ms,
                    end_ms = EXCLUDED.end_ms
                """
            )
        else:
            # SQLite syntax
            sql = text(
                """
                INSERT OR REPLACE INTO object_labels
                    (artifact_id, asset_id, label, confidence, start_ms, end_ms)
                VALUES (
                    :artifact_id, :asset_id, :label, :confidence,
                    :start_ms, :end_ms
                )
                """
            )

        self.session.execute(
            sql,
            {
                "artifact_id": artifact.artifact_id,
                "asset_id": artifact.asset_id,
                "label": label,
                "confidence": confidence,
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms,
            },
        )

        logger.debug(
            f"Synced object.detection artifact {artifact.artifact_id} "
            f"to object_labels projection (label={label})"
        )

    def _sync_face_clusters(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize face.detection artifact to face_clusters projection.

        Args:
            artifact: The face.detection artifact to synchronize
        """
        # Parse payload to extract cluster_id and confidence
        payload = json.loads(artifact.payload_json)
        cluster_id = payload.get("cluster_id")
        confidence = payload.get("confidence", 0.0)

        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        if is_postgresql:
            # PostgreSQL syntax
            sql = text(
                """
                INSERT INTO face_clusters
                    (artifact_id, asset_id, cluster_id, confidence,
                     start_ms, end_ms)
                VALUES (:artifact_id, :asset_id, :cluster_id, :confidence,
                        :start_ms, :end_ms)
                ON CONFLICT (artifact_id) DO UPDATE
                SET asset_id = EXCLUDED.asset_id,
                    cluster_id = EXCLUDED.cluster_id,
                    confidence = EXCLUDED.confidence,
                    start_ms = EXCLUDED.start_ms,
                    end_ms = EXCLUDED.end_ms
                """
            )
        else:
            # SQLite syntax
            sql = text(
                """
                INSERT OR REPLACE INTO face_clusters
                    (artifact_id, asset_id, cluster_id, confidence,
                     start_ms, end_ms)
                VALUES (:artifact_id, :asset_id, :cluster_id, :confidence,
                        :start_ms, :end_ms)
                """
            )

        self.session.execute(
            sql,
            {
                "artifact_id": artifact.artifact_id,
                "asset_id": artifact.asset_id,
                "cluster_id": cluster_id,
                "confidence": confidence,
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms,
            },
        )

        logger.debug(
            f"Synced face.detection artifact {artifact.artifact_id} "
            f"to face_clusters projection "
            f"(cluster_id={cluster_id})"
        )

    def _sync_ocr_fts(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize ocr.text artifact to FTS projection.

        Args:
            artifact: The ocr.text artifact to synchronize
        """
        # Parse payload to extract text
        payload = json.loads(artifact.payload_json)
        ocr_text = payload.get("text", "")

        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        if is_postgresql:
            # PostgreSQL: Insert into ocr_fts table
            # The tsvector column is automatically computed
            sql = text(
                """
                INSERT INTO ocr_fts
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
                INSERT OR REPLACE INTO ocr_fts_metadata
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
                INSERT INTO ocr_fts
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
                "text": ocr_text,
            },
        )

        logger.debug(
            f"Synced ocr.text artifact {artifact.artifact_id} to FTS projection"
        )

    def _sync_video_metadata(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize video.metadata artifact to video_locations projection.

        Extracts GPS coordinates from metadata payload and creates an entry
        in the video_locations projection table for geo-spatial queries.
        Performs reverse geocoding to populate country, state, and city fields.

        Args:
            artifact: The video.metadata artifact to synchronize

        Raises:
            ProjectionSyncError: If GPS coordinates are invalid or sync fails
        """
        # Parse payload to extract GPS coordinates
        payload = json.loads(artifact.payload_json)
        latitude = payload.get("latitude")
        longitude = payload.get("longitude")
        altitude = payload.get("altitude")

        # Only create projection entry if GPS coordinates exist
        if latitude is None or longitude is None:
            logger.debug(
                f"No GPS coordinates in metadata artifact {artifact.artifact_id}, "
                f"skipping video_locations projection"
            )
            return

        # Validate GPS coordinates
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            altitude = float(altitude) if altitude is not None else None

            # Validate latitude range: -90 to 90
            if not (-90 <= latitude <= 90):
                raise ValueError(
                    f"Invalid latitude {latitude}: must be between -90 and 90"
                )

            # Validate longitude range: -180 to 180
            if not (-180 <= longitude <= 180):
                raise ValueError(
                    f"Invalid longitude {longitude}: must be between -180 and 180"
                )

        except (ValueError, TypeError) as e:
            logger.error(
                f"Invalid GPS coordinates in metadata artifact "
                f"{artifact.artifact_id}: {e}"
            )
            raise ProjectionSyncError(f"Invalid GPS coordinates: {e}") from e

        # Perform reverse geocoding to get location names
        from src.services.reverse_geocoding_service import ReverseGeocodingService

        logger.info(f"🔄 Starting reverse geocoding for {artifact.artifact_id}")
        geocoding_service = ReverseGeocodingService()
        location_info = geocoding_service.get_location_info(latitude, longitude)
        country = location_info.get("country")
        state = location_info.get("state")
        city = location_info.get("city")
        logger.info(
            f"✓ Geocoding complete: country={country}, state={state}, city={city}"
        )

        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        if is_postgresql:
            # PostgreSQL syntax - UPSERT on video_id
            # Each video has only one metadata set, so we overwrite on re-extraction
            sql = text(
                """
                INSERT INTO video_locations
                    (artifact_id, video_id, latitude, longitude, altitude,
                     country, state, city)
                VALUES (:artifact_id, :video_id, :latitude, :longitude, :altitude,
                        :country, :state, :city)
                ON CONFLICT (video_id) DO UPDATE
                SET artifact_id = EXCLUDED.artifact_id,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    altitude = EXCLUDED.altitude,
                    country = EXCLUDED.country,
                    state = EXCLUDED.state,
                    city = EXCLUDED.city
                """
            )
        else:
            # SQLite syntax
            sql = text(
                """
                INSERT OR REPLACE INTO video_locations
                    (artifact_id, video_id, latitude, longitude, altitude,
                     country, state, city)
                VALUES (:artifact_id, :video_id, :latitude, :longitude, :altitude,
                        :country, :state, :city)
                """
            )

        self.session.execute(
            sql,
            {
                "artifact_id": artifact.artifact_id,
                "video_id": artifact.asset_id,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "country": country,
                "state": state,
                "city": city,
            },
        )

        logger.debug(
            f"Synced video.metadata artifact {artifact.artifact_id} "
            f"to video_locations projection "
            f"(lat={latitude}, lon={longitude}, "
            f"country={country}, state={state}, city={city})"
        )
