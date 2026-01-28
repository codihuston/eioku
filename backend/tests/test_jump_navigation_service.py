"""Tests for JumpNavigationService."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database.models import Video as VideoEntity
from src.domain.artifacts import ArtifactEnvelope, SelectionPolicy
from src.domain.schema_initialization import register_all_schemas
from src.domain.schema_registry import SchemaRegistry
from src.repositories.artifact_repository import SqlArtifactRepository
from src.repositories.selection_policy_manager import SelectionPolicyManager
from src.services.jump_navigation_service import JumpNavigationService


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


@pytest.fixture
def artifact_repo(session, schema_registry):
    """Create artifact repository instance with mocked projection sync."""
    # Mock the projection sync service to avoid FTS table errors
    mock_projection_sync = MagicMock()
    mock_projection_sync.sync_artifact = MagicMock()
    return SqlArtifactRepository(session, schema_registry, mock_projection_sync)


@pytest.fixture
def policy_manager(session):
    """Create selection policy manager instance."""
    return SelectionPolicyManager(session)


@pytest.fixture
def jump_service(artifact_repo, policy_manager):
    """Create jump navigation service instance."""
    return JumpNavigationService(artifact_repo, policy_manager)


@pytest.fixture
def test_video(session):
    """Create a test video entity."""
    video = VideoEntity(
        video_id="test_video_1",
        file_path="/test/video.mp4",
        filename="video.mp4",
        last_modified=datetime.now(),
        status="completed",
    )
    session.add(video)
    session.commit()
    return video


def create_transcript_artifact(
    artifact_id, asset_id, start_ms, end_ms, text, confidence=0.9, run_id="run_1"
):
    """Helper to create transcript artifact."""
    payload = {
        "text": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": confidence,
    }
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


def create_object_artifact(
    artifact_id,
    asset_id,
    start_ms,
    end_ms,
    label,
    confidence=0.9,
    run_id="run_1",
    frame_number=0,
):
    """Helper to create object detection artifact."""
    payload = {
        "label": label,
        "confidence": confidence,
        "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        "frame_number": frame_number,
    }
    return ArtifactEnvelope(
        artifact_id=artifact_id,
        asset_id=asset_id,
        artifact_type="object.detection",
        schema_version=1,
        span_start_ms=start_ms,
        span_end_ms=end_ms,
        payload_json=json.dumps(payload),
        producer="yolo",
        producer_version="8.0.0",
        model_profile="balanced",
        config_hash="xyz789",
        input_hash="uvw012",
        run_id=run_id,
        created_at=datetime.now(),
    )


def create_face_artifact(
    artifact_id,
    asset_id,
    start_ms,
    end_ms,
    cluster_id,
    confidence=0.9,
    run_id="run_1",
    frame_number=0,
):
    """Helper to create face detection artifact."""
    payload = {
        "confidence": confidence,
        "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        "cluster_id": cluster_id,
        "frame_number": frame_number,
    }
    return ArtifactEnvelope(
        artifact_id=artifact_id,
        asset_id=asset_id,
        artifact_type="face.detection",
        schema_version=1,
        span_start_ms=start_ms,
        span_end_ms=end_ms,
        payload_json=json.dumps(payload),
        producer="yolo-face",
        producer_version="8.0.0",
        model_profile="balanced",
        config_hash="face123",
        input_hash="face456",
        run_id=run_id,
        created_at=datetime.now(),
    )


def create_scene_artifact(
    artifact_id, asset_id, start_ms, end_ms, scene_index, run_id="run_1"
):
    """Helper to create scene artifact."""
    payload = {
        "scene_index": scene_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
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


def test_jump_next_transcript(jump_service, artifact_repo, test_video):
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

    # Jump from 500ms (should get artifact2)
    result = jump_service.jump_next(
        test_video.video_id, "transcript.segment", from_ms=500
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 1000
    assert result["jump_to"]["end_ms"] == 2000
    assert "t2" in result["artifact_ids"]


def test_jump_next_no_match(jump_service, artifact_repo, test_video):
    """Test jumping next when no artifacts exist after timestamp."""
    artifact1 = create_transcript_artifact(
        "t1", test_video.video_id, 0, 1000, "Only segment"
    )
    artifact_repo.create(artifact1)

    # Jump from 2000ms (no artifacts after this)
    result = jump_service.jump_next(
        test_video.video_id, "transcript.segment", from_ms=2000
    )

    assert result is None


def test_jump_prev_transcript(jump_service, artifact_repo, test_video):
    """Test jumping to previous transcript segment."""
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

    # Jump from 2500ms (should get artifact2)
    result = jump_service.jump_prev(
        test_video.video_id, "transcript.segment", from_ms=2500
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 1000
    assert result["jump_to"]["end_ms"] == 2000
    assert "t2" in result["artifact_ids"]


def test_jump_prev_no_match(jump_service, artifact_repo, test_video):
    """Test jumping prev when no artifacts exist before timestamp."""
    artifact1 = create_transcript_artifact(
        "t1", test_video.video_id, 1000, 2000, "Only segment"
    )
    artifact_repo.create(artifact1)

    # Jump from 500ms (no artifacts before this)
    result = jump_service.jump_prev(
        test_video.video_id, "transcript.segment", from_ms=500
    )

    assert result is None


def test_jump_next_with_label_filter(jump_service, artifact_repo, test_video):
    """Test jumping next with label filtering for objects."""
    # Create object artifacts with different labels
    artifact1 = create_object_artifact(
        "o1", test_video.video_id, 0, 100, "dog", frame_number=0
    )
    artifact2 = create_object_artifact(
        "o2", test_video.video_id, 100, 200, "cat", frame_number=10
    )
    artifact3 = create_object_artifact(
        "o3", test_video.video_id, 200, 300, "dog", frame_number=20
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump to next "dog" from 50ms (should skip cat and get second dog)
    result = jump_service.jump_next(
        test_video.video_id, "object.detection", from_ms=50, label="dog"
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 200
    assert "o3" in result["artifact_ids"]


def test_jump_next_with_cluster_filter(jump_service, artifact_repo, test_video):
    """Test jumping next with cluster filtering for faces."""
    # Create face artifacts with different clusters
    artifact1 = create_face_artifact(
        "f1", test_video.video_id, 0, 100, "cluster_a", frame_number=0
    )
    artifact2 = create_face_artifact(
        "f2", test_video.video_id, 100, 200, "cluster_b", frame_number=10
    )
    artifact3 = create_face_artifact(
        "f3", test_video.video_id, 200, 300, "cluster_a", frame_number=20
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump to next cluster_a from 50ms (should skip cluster_b)
    result = jump_service.jump_next(
        test_video.video_id, "face.detection", from_ms=50, cluster_id="cluster_a"
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 200
    assert "f3" in result["artifact_ids"]


def test_jump_next_with_confidence_filter(jump_service, artifact_repo, test_video):
    """Test jumping next with minimum confidence filtering."""
    # Create artifacts with different confidence levels
    artifact1 = create_object_artifact(
        "o1", test_video.video_id, 0, 100, "dog", confidence=0.5, frame_number=0
    )
    artifact2 = create_object_artifact(
        "o2", test_video.video_id, 100, 200, "dog", confidence=0.7, frame_number=10
    )
    artifact3 = create_object_artifact(
        "o3", test_video.video_id, 200, 300, "dog", confidence=0.9, frame_number=20
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump with min_confidence=0.8 (should skip first two)
    result = jump_service.jump_next(
        test_video.video_id, "object.detection", from_ms=0, min_confidence=0.8
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 200
    assert "o3" in result["artifact_ids"]


def test_jump_prev_with_label_filter(jump_service, artifact_repo, test_video):
    """Test jumping prev with label filtering."""
    # Create object artifacts with different labels
    artifact1 = create_object_artifact(
        "o1", test_video.video_id, 0, 100, "dog", frame_number=0
    )
    artifact2 = create_object_artifact(
        "o2", test_video.video_id, 100, 200, "cat", frame_number=10
    )
    artifact3 = create_object_artifact(
        "o3", test_video.video_id, 200, 300, "dog", frame_number=20
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump to prev "dog" from 250ms (should skip cat and get first dog)
    result = jump_service.jump_prev(
        test_video.video_id, "object.detection", from_ms=250, label="dog"
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 0
    assert "o1" in result["artifact_ids"]


def test_jump_next_scene(jump_service, artifact_repo, test_video):
    """Test jumping to next scene."""
    # Create scene artifacts
    artifact1 = create_scene_artifact("s1", test_video.video_id, 0, 5000, 1)
    artifact2 = create_scene_artifact("s2", test_video.video_id, 5000, 10000, 2)
    artifact3 = create_scene_artifact("s3", test_video.video_id, 10000, 15000, 3)

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump from 3000ms (should get scene 2)
    result = jump_service.jump_next(test_video.video_id, "scene", from_ms=3000)

    assert result is not None
    assert result["jump_to"]["start_ms"] == 5000
    assert "s2" in result["artifact_ids"]


def test_jump_with_selection_policy(
    jump_service, artifact_repo, policy_manager, test_video
):
    """Test jumping with selection policy."""
    # Create artifacts from different runs
    artifact1 = create_transcript_artifact(
        "t1", test_video.video_id, 0, 1000, "Run 1", run_id="run_1"
    )
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Run 2", run_id="run_2"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)

    # Set policy to use run_2 only
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="pinned",
        pinned_run_id="run_2",
    )
    policy_manager.set_policy(policy)

    # Jump from 0ms (should only see run_2 artifacts, which starts at 1000ms)
    result = jump_service.jump_next(
        test_video.video_id, "transcript.segment", from_ms=0
    )

    # Should find artifact2 from run_2, not artifact1 from run_1
    assert result is not None
    assert result["jump_to"]["start_ms"] == 1000
    assert "t2" in result["artifact_ids"]


def test_jump_next_boundary_case(jump_service, artifact_repo, test_video):
    """Test jumping next from exact artifact start time."""
    artifact1 = create_transcript_artifact("t1", test_video.video_id, 0, 1000, "First")
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Second"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)

    # Jump from exactly 1000ms (should get artifact2)
    result = jump_service.jump_next(
        test_video.video_id, "transcript.segment", from_ms=1000
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 1000
    assert "t2" in result["artifact_ids"]


def test_jump_prev_boundary_case(jump_service, artifact_repo, test_video):
    """Test jumping prev from within an artifact doesn't return that artifact."""
    artifact1 = create_transcript_artifact("t1", test_video.video_id, 0, 1000, "First")
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1000, 2000, "Second"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)

    # Jump from 1500ms (within artifact2) should get artifact1
    result = jump_service.jump_prev(
        test_video.video_id, "transcript.segment", from_ms=1500
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 0
    assert "t1" in result["artifact_ids"]


def test_jump_with_multiple_filters(jump_service, artifact_repo, test_video):
    """Test jumping with multiple filters (label + confidence)."""
    # Create artifacts with different labels and confidence
    artifact1 = create_object_artifact(
        "o1", test_video.video_id, 0, 100, "dog", confidence=0.6, frame_number=0
    )
    artifact2 = create_object_artifact(
        "o2", test_video.video_id, 100, 200, "dog", confidence=0.9, frame_number=10
    )
    artifact3 = create_object_artifact(
        "o3", test_video.video_id, 200, 300, "cat", confidence=0.95, frame_number=20
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump with label="dog" and min_confidence=0.8
    result = jump_service.jump_next(
        test_video.video_id,
        "object.detection",
        from_ms=0,
        label="dog",
        min_confidence=0.8,
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 100
    assert "o2" in result["artifact_ids"]


def test_jump_empty_database(jump_service, test_video):
    """Test jumping when no artifacts exist."""
    result = jump_service.jump_next(
        test_video.video_id, "transcript.segment", from_ms=0
    )
    assert result is None

    result = jump_service.jump_prev(
        test_video.video_id, "transcript.segment", from_ms=1000
    )
    assert result is None


def test_filter_artifacts_with_invalid_payload(jump_service, artifact_repo, test_video):
    """Test that artifacts with invalid payloads are skipped during filtering."""
    # Create a valid artifact
    artifact1 = create_object_artifact(
        "o1", test_video.video_id, 0, 100, "dog", frame_number=0
    )
    artifact_repo.create(artifact1)

    # Manually create an artifact with invalid JSON (this bypasses validation)
    # In practice, this shouldn't happen, but we test defensive coding
    invalid_artifact = ArtifactEnvelope(
        artifact_id="invalid",
        asset_id=test_video.video_id,
        artifact_type="object.detection",
        schema_version=1,
        span_start_ms=100,
        span_end_ms=200,
        payload_json="invalid json",  # Invalid JSON
        producer="test",
        producer_version="1.0.0",
        model_profile="balanced",
        config_hash="test",
        input_hash="test",
        run_id="run_1",
        created_at=datetime.now(),
    )

    # The filter should handle this gracefully
    artifacts = [artifact1, invalid_artifact]
    filtered = jump_service._filter_artifacts(artifacts, None, None, 0.0)

    # Should only return the valid artifact
    assert len(filtered) == 1
    assert filtered[0].artifact_id == "o1"


def test_jump_next_returns_earliest_match(jump_service, artifact_repo, test_video):
    """Test that jump_next returns the earliest matching artifact."""
    # Create multiple artifacts after the from_ms timestamp
    artifact1 = create_transcript_artifact(
        "t1", test_video.video_id, 1000, 2000, "First"
    )
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 1500, 2500, "Second"
    )
    artifact3 = create_transcript_artifact(
        "t3", test_video.video_id, 2000, 3000, "Third"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump from 500ms (should get artifact1, the earliest)
    result = jump_service.jump_next(
        test_video.video_id, "transcript.segment", from_ms=500
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 1000
    assert "t1" in result["artifact_ids"]


def test_jump_prev_returns_latest_match(jump_service, artifact_repo, test_video):
    """Test that jump_prev returns the latest matching artifact before from_ms."""
    # Create multiple artifacts before the from_ms timestamp
    artifact1 = create_transcript_artifact("t1", test_video.video_id, 0, 1000, "First")
    artifact2 = create_transcript_artifact(
        "t2", test_video.video_id, 500, 1500, "Second"
    )
    artifact3 = create_transcript_artifact(
        "t3", test_video.video_id, 1000, 2000, "Third"
    )

    artifact_repo.create(artifact1)
    artifact_repo.create(artifact2)
    artifact_repo.create(artifact3)

    # Jump from 2500ms (should get artifact3, the latest before 2500ms)
    result = jump_service.jump_prev(
        test_video.video_id, "transcript.segment", from_ms=2500
    )

    assert result is not None
    assert result["jump_to"]["start_ms"] == 1000
    assert "t3" in result["artifact_ids"]
