"""Tests for artifact API endpoints."""

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import Base
from src.database.models import Video as VideoEntity
from src.domain.artifacts import ArtifactEnvelope
from src.domain.schema_initialization import register_all_schemas
from src.domain.schema_registry import SchemaRegistry
from src.repositories.artifact_repository import SqlArtifactRepository
from src.repositories.selection_policy_manager import SelectionPolicyManager
from src.services.find_within_video_service import FindWithinVideoService
from src.services.jump_navigation_service import JumpNavigationService


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing."""
    # Use check_same_thread=False to allow SQLite to be used across threads in tests
    # Use StaticPool to share connection across threads
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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


@pytest.fixture
def artifact_repo(session, schema_registry):
    """Create artifact repository instance."""
    return SqlArtifactRepository(session, schema_registry)


@pytest.fixture
def test_video(session):
    """Create a test video entity."""
    video = VideoEntity(
        video_id="test_video_api",
        file_path="/test/api_video.mp4",
        filename="api_video.mp4",
        last_modified=datetime.now(),
        status="completed",
    )
    session.add(video)
    session.commit()
    return video


@pytest.fixture
def client(engine, session):
    """Create test client with in-memory database."""
    from unittest.mock import Mock

    from fastapi import FastAPI

    from src.api.artifact_controller import (
        get_artifact_repository,
        get_find_within_video_service,
        get_jump_navigation_service,
        get_selection_policy_manager,
    )
    from src.api.artifact_controller import (
        router as artifact_router,
    )
    from src.database.connection import get_db
    from src.services.projection_sync_service import ProjectionSyncService

    # Create a mock projection sync service that does nothing
    mock_projection_sync = Mock(spec=ProjectionSyncService)
    mock_projection_sync.sync_artifact = Mock()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    def override_jump_service():
        from src.domain.schema_registry import SchemaRegistry

        artifact_repo = SqlArtifactRepository(
            session, SchemaRegistry, mock_projection_sync
        )
        policy_manager = SelectionPolicyManager(session)
        return JumpNavigationService(artifact_repo, policy_manager)

    def override_find_service():
        policy_manager = SelectionPolicyManager(session)
        return FindWithinVideoService(session, policy_manager)

    def override_artifact_repo():
        from src.domain.schema_registry import SchemaRegistry

        return SqlArtifactRepository(session, SchemaRegistry, mock_projection_sync)

    def override_policy_manager():
        return SelectionPolicyManager(session)

    # Create app without lifespan to avoid startup issues in tests
    app = FastAPI()
    app.include_router(artifact_router, prefix="/v1")
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_jump_navigation_service] = override_jump_service
    app.dependency_overrides[get_find_within_video_service] = override_find_service
    app.dependency_overrides[get_artifact_repository] = override_artifact_repo
    app.dependency_overrides[get_selection_policy_manager] = override_policy_manager

    return TestClient(app)


def create_transcript_artifact(
    artifact_id, asset_id, start_ms, end_ms, text, confidence=0.9, run_id="run_1"
):
    """Helper to create transcript artifact."""
    payload = {"text": text, "confidence": confidence, "language": "en"}
    return ArtifactEnvelope(
        artifact_id=artifact_id,
        asset_id=asset_id,
        artifact_type="transcript.segment",
        schema_version=1,
        span_start_ms=start_ms,
        span_end_ms=end_ms,
        payload_json=json.dumps(payload),
        producer="whisper",
        producer_version="3.0.0",
        model_profile="balanced",
        config_hash="abc123",
        input_hash="def456",
        run_id=run_id,
        created_at=datetime.now(),
    )


def create_scene_artifact(
    artifact_id, asset_id, start_ms, end_ms, scene_index, run_id="run_1"
):
    """Helper to create scene artifact."""
    payload = {
        "scene_index": scene_index,
        "method": "content",
        "score": 0.8,
        "frame_number": scene_index * 100,
    }
    return ArtifactEnvelope(
        artifact_id=artifact_id,
        asset_id=asset_id,
        artifact_type="scene",
        schema_version=1,
        span_start_ms=start_ms,
        span_end_ms=end_ms,
        payload_json=json.dumps(payload),
        producer="pyscenedetect",
        producer_version="0.6.0",
        model_profile="balanced",
        config_hash="scene123",
        input_hash="scene456",
        run_id=run_id,
        created_at=datetime.now(),
    )


def test_jump_next_transcript(client, artifact_repo, test_video):
    """Test jumping to next transcript segment."""
    # Create transcript artifacts
    artifact1 = create_transcript_artifact(
        "t1", test_video.video_id, 0, 1000, "First segment"
    )
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Second segment"
    )
    artifact3 = create_transcript_artifact(
        "t3", test_video.video_id, 2000, 3000, "Third segment"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    response = client.get(
        f"/v1/videos/{test_video.video_id}/jump",
        params={
            "kind": "transcript",
            "direction": "next",
            "from_ms": 500,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "jump_to" in data
    assert data["jump_to"]["start_ms"] == 1000
    assert data["jump_to"]["end_ms"] == 2000
    assert len(data["artifact_ids"]) == 1


def test_jump_prev_transcript(client, artifact_repo, test_video):
    """Test jumping to previous transcript segment."""
    artifact1 = create_transcript_artifact(
        "t1", test_video.video_id, 0, 1000, "First segment"
    )
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Second segment"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)

    response = client.get(
        f"/v1/videos/{test_video.video_id}/jump",
        params={
            "kind": "transcript",
            "direction": "prev",
            "from_ms": 2500,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "jump_to" in data
    assert data["jump_to"]["start_ms"] == 1000
    assert data["jump_to"]["end_ms"] == 2000


def test_jump_next_scene(client, artifact_repo, test_video):
    """Test jumping to next scene."""
    artifact1 = create_scene_artifact("s1", test_video.video_id, 0, 5000, 1)
    artifact2 = create_scene_artifact("s2", test_video.video_id, 5000, 10000, 2)

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)

    response = client.get(
        f"/v1/videos/{test_video.video_id}/jump",
        params={
            "kind": "scene",
            "direction": "next",
            "from_ms": 500,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["jump_to"]["start_ms"] == 5000
    assert data["jump_to"]["end_ms"] == 10000


def test_jump_not_found(client, artifact_repo, test_video):
    """Test jump when no artifact found."""
    artifact1 = create_transcript_artifact(
        "t1", test_video.video_id, 0, 1000, "Only segment"
    )
    artifact_repo.create(artifact1)

    response = client.get(
        f"/v1/videos/{test_video.video_id}/jump",
        params={
            "kind": "transcript",
            "direction": "next",
            "from_ms": 10000,  # Beyond all artifacts
        },
    )

    assert response.status_code == 404
    assert (
        "artifact found" in response.json()["detail"].lower()
        or "not found" in response.json()["detail"].lower()
    )


def test_jump_invalid_kind(client, test_video):
    """Test jump with invalid kind parameter."""
    response = client.get(
        f"/v1/videos/{test_video.video_id}/jump",
        params={
            "kind": "invalid",
            "direction": "next",
            "from_ms": 0,
        },
    )

    assert response.status_code == 400
    assert "Invalid kind" in response.json()["detail"]


def test_jump_invalid_direction(client, test_video):
    """Test jump with invalid direction parameter."""
    response = client.get(
        f"/v1/videos/{test_video.video_id}/jump",
        params={
            "kind": "transcript",
            "direction": "invalid",
            "from_ms": 0,
        },
    )

    assert response.status_code == 400
    assert "Invalid direction" in response.json()["detail"]


def test_get_artifacts(client, artifact_repo, test_video):
    """Test getting all artifacts for a video."""
    # Create some artifacts
    artifact1 = create_transcript_artifact("t1", test_video.video_id, 0, 1000, "First")
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Second"
    )
    artifact3 = create_scene_artifact("s1", test_video.video_id, 0, 5000, 1)

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    response = client.get(f"/v1/videos/{test_video.video_id}/artifacts")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert all("artifact_id" in item for item in data)
    assert all("payload" in item for item in data)


def test_get_artifacts_filtered_by_type(client, artifact_repo, test_video):
    """Test getting artifacts filtered by type."""
    artifact1 = create_transcript_artifact("t1", test_video.video_id, 0, 1000, "First")
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Second"
    )
    artifact3 = create_scene_artifact("s1", test_video.video_id, 0, 5000, 1)

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    response = client.get(
        f"/v1/videos/{test_video.video_id}/artifacts",
        params={"type": "transcript.segment"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(item["artifact_type"] == "transcript.segment" for item in data)


def test_get_artifacts_filtered_by_time_range(client, artifact_repo, test_video):
    """Test getting artifacts filtered by time range."""
    artifact1 = create_transcript_artifact("t1", test_video.video_id, 0, 1000, "First")
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Second"
    )
    artifact3 = create_transcript_artifact(
        "t3", test_video.video_id, 2000, 3000, "Third"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    response = client.get(
        f"/v1/videos/{test_video.video_id}/artifacts",
        params={
            "from_ms": 1000,
            "to_ms": 2000,
        },
    )

    assert response.status_code == 200
    data = response.json()
    # Should get artifacts that start >= 1000 and end <= 2000
    assert len(data) >= 1
    for item in data:
        assert item["span_start_ms"] >= 1000
        assert item["span_end_ms"] <= 2000


def test_get_artifacts_empty_result(client, test_video):
    """Test getting artifacts when none exist."""
    response = client.get(f"/v1/videos/{test_video.video_id}/artifacts")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0


def test_get_profiles(client, artifact_repo, test_video):
    """Test getting available profiles for a video and artifact type."""
    # Create artifacts with different profiles
    # Fast profile - 3 artifacts
    for i in range(3):
        artifact = create_transcript_artifact(
            f"t_fast_{i}",
            test_video.video_id,
            i * 1000,
            (i + 1) * 1000,
            f"Fast segment {i}",
        )
        artifact.model_profile = "fast"
        artifact.run_id = "run_fast"
        artifact_repo.create(artifact)

    # Balanced profile - 5 artifacts
    for i in range(5):
        artifact = create_transcript_artifact(
            f"t_balanced_{i}",
            test_video.video_id,
            i * 1000,
            (i + 1) * 1000,
            f"Balanced segment {i}",
        )
        artifact.model_profile = "balanced"
        artifact.run_id = "run_balanced"
        artifact_repo.create(artifact)

    # High quality profile - 2 artifacts
    for i in range(2):
        artifact = create_transcript_artifact(
            f"t_hq_{i}",
            test_video.video_id,
            i * 1000,
            (i + 1) * 1000,
            f"HQ segment {i}",
        )
        artifact.model_profile = "high_quality"
        artifact.run_id = "run_hq"
        artifact_repo.create(artifact)

    # Test the profiles endpoint
    response = client.get(
        f"/v1/videos/{test_video.video_id}/profiles",
        params={"artifact_type": "transcript.segment"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["video_id"] == test_video.video_id
    assert data["artifact_type"] == "transcript.segment"
    assert len(data["profiles"]) == 3

    # Check profiles are sorted alphabetically
    profiles = {p["profile"]: p for p in data["profiles"]}
    assert "fast" in profiles
    assert "balanced" in profiles
    assert "high_quality" in profiles

    # Check artifact counts
    assert profiles["fast"]["artifact_count"] == 3
    assert profiles["balanced"]["artifact_count"] == 5
    assert profiles["high_quality"]["artifact_count"] == 2

    # Check run IDs
    assert profiles["fast"]["run_ids"] == ["run_fast"]
    assert profiles["balanced"]["run_ids"] == ["run_balanced"]
    assert profiles["high_quality"]["run_ids"] == ["run_hq"]


def test_get_profiles_empty(client, test_video):
    """Test getting profiles when no artifacts exist."""
    response = client.get(
        f"/v1/videos/{test_video.video_id}/profiles",
        params={"artifact_type": "transcript.segment"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["video_id"] == test_video.video_id
    assert data["artifact_type"] == "transcript.segment"
    assert data["profiles"] == []
