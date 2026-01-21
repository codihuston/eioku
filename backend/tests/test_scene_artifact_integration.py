"""Integration tests for scene artifact creation and projection sync."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.domain.artifacts import ArtifactEnvelope
from src.domain.schema_initialization import register_all_schemas
from src.domain.schema_registry import SchemaRegistry
from src.domain.schemas.scene_v1 import SceneV1
from src.repositories.artifact_repository import SqlArtifactRepository
from src.services.projection_sync_service import ProjectionSyncService
from src.services.scene_detection_service import SceneDetectionService


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create database session for testing."""
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture(scope="session")
def schema_registry():
    """Create and initialize schema registry once for all tests."""
    register_all_schemas()
    return SchemaRegistry


class TestSceneArtifactIntegration:
    """Test scene artifact creation and projection synchronization."""

    def test_scene_artifact_creation(self, session, schema_registry):
        """Test that scene detection creates valid artifacts."""
        # Create repositories
        projection_sync = ProjectionSyncService(session)
        artifact_repo = SqlArtifactRepository(session, schema_registry, projection_sync)

        # Create a scene artifact
        scene_payload = SceneV1(
            scene_index=0, method="content", score=0.85, frame_number=120
        )

        artifact = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id="test_video_1",
            artifact_type="scene",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=5000,
            payload_json=scene_payload.model_dump_json(),
            producer="ffmpeg",
            producer_version="1.0.0",
            model_profile="balanced",
            config_hash="test_config_hash",
            input_hash="test_input_hash",
            run_id="test_run_1",
            created_at=datetime.utcnow(),
        )

        # Create the artifact
        created_artifact = artifact_repo.create(artifact)

        # Verify artifact was created
        assert created_artifact.artifact_id == artifact.artifact_id
        assert created_artifact.artifact_type == "scene"

        # Verify projection was synchronized
        result = session.execute(
            text(
                """
                SELECT artifact_id, asset_id, scene_index, start_ms, end_ms
                FROM scene_ranges
                WHERE artifact_id = :artifact_id
                """
            ),
            {"artifact_id": artifact.artifact_id},
        ).fetchone()

        assert result is not None
        assert result[0] == artifact.artifact_id
        assert result[1] == "test_video_1"
        assert result[2] == 0  # scene_index
        assert result[3] == 0  # start_ms
        assert result[4] == 5000  # end_ms

    def test_scene_detection_service_creates_artifacts(self):
        """Test that scene detection service creates artifact envelopes."""
        service = SceneDetectionService(threshold=0.4, min_scene_len=0.6)

        # Mock the Path.exists check and FFmpeg calls
        with patch.object(Path, "exists", return_value=True), patch.object(
            service, "_get_video_duration", return_value=10.0
        ), patch.object(service, "_detect_scene_changes", return_value=[3.0, 7.0]):
            artifacts = service.detect_scenes(
                video_path="/fake/path.mp4",
                video_id="test_video",
                run_id="test_run",
                model_profile="balanced",
            )

            # Should create 3 scenes: [0-3s], [3-7s], [7-10s]
            assert len(artifacts) == 3

            # Verify first artifact
            assert artifacts[0].artifact_type == "scene"
            assert artifacts[0].span_start_ms == 0
            assert artifacts[0].span_end_ms == 3000
            assert artifacts[0].producer == "ffmpeg"
            assert artifacts[0].model_profile == "balanced"

            # Verify payload
            payload = json.loads(artifacts[0].payload_json)
            assert payload["scene_index"] == 0
            assert payload["method"] == "content"

    def test_multiple_scene_artifacts_projection(self, session, schema_registry):
        """Test that multiple scene artifacts are correctly projected."""
        projection_sync = ProjectionSyncService(session)
        artifact_repo = SqlArtifactRepository(session, schema_registry, projection_sync)

        # Create multiple scene artifacts
        run_id = str(uuid.uuid4())
        asset_id = "test_video_multi"

        for i in range(3):
            scene_payload = SceneV1(
                scene_index=i,
                method="content",
                score=0.8 + i * 0.05,
                frame_number=i * 100,
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=asset_id,
                artifact_type="scene",
                schema_version=1,
                span_start_ms=i * 5000,
                span_end_ms=(i + 1) * 5000,
                payload_json=scene_payload.model_dump_json(),
                producer="ffmpeg",
                producer_version="1.0.0",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify all scenes are in projection
        result = session.execute(
            text(
                """
                SELECT COUNT(*), MIN(scene_index), MAX(scene_index)
                FROM scene_ranges
                WHERE asset_id = :asset_id
                """
            ),
            {"asset_id": asset_id},
        ).fetchone()

        assert result[0] == 3  # count
        assert result[1] == 0  # min scene_index
        assert result[2] == 2  # max scene_index

    def test_scene_ranges_query_by_asset_and_index(self, session, schema_registry):
        """Test querying scene_ranges by asset_id and scene_index."""
        projection_sync = ProjectionSyncService(session)
        artifact_repo = SqlArtifactRepository(session, schema_registry, projection_sync)

        # Create a scene artifact
        scene_payload = SceneV1(
            scene_index=5, method="content", score=0.9, frame_number=500
        )

        artifact = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id="test_video_query",
            artifact_type="scene",
            schema_version=1,
            span_start_ms=10000,
            span_end_ms=15000,
            payload_json=scene_payload.model_dump_json(),
            producer="ffmpeg",
            producer_version="1.0.0",
            model_profile="balanced",
            config_hash="test_config",
            input_hash="test_input",
            run_id="test_run",
            created_at=datetime.utcnow(),
        )

        artifact_repo.create(artifact)

        # Query by asset_id and scene_index
        result = session.execute(
            text(
                """
                SELECT artifact_id, start_ms, end_ms
                FROM scene_ranges
                WHERE asset_id = :asset_id AND scene_index = :scene_index
                """
            ),
            {"asset_id": "test_video_query", "scene_index": 5},
        ).fetchone()

        assert result is not None
        assert result[0] == artifact.artifact_id
        assert result[1] == 10000
        assert result[2] == 15000
