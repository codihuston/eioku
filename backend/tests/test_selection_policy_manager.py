"""Tests for SelectionPolicyManager."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Video
from src.domain.artifacts import SelectionPolicy
from src.repositories.selection_policy_manager import SelectionPolicyManager


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create database session for testing."""
    session_local = sessionmaker(bind=engine)
    session = session_local()
    yield session
    session.close()


@pytest.fixture
def test_video(session):
    """Create a test video."""
    video = Video(
        video_id="test_video_1",
        file_path="/path/to/video.mp4",
        filename="video.mp4",
        duration=120.0,
        file_size=1024000,
        last_modified=datetime.utcnow(),
    )
    session.add(video)
    session.commit()
    return video


@pytest.fixture
def manager(session):
    """Create SelectionPolicyManager instance."""
    return SelectionPolicyManager(session)


def test_get_policy_not_found(manager):
    """Test getting a policy that doesn't exist returns None."""
    policy = manager.get_policy("nonexistent_video", "transcript.segment")
    assert policy is None


def test_set_policy_creates_new(manager, test_video):
    """Test setting a new policy creates it in the database."""
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="profile",
        preferred_profile="high_quality",
    )

    result = manager.set_policy(policy)

    assert result.asset_id == test_video.video_id
    assert result.artifact_type == "transcript.segment"
    assert result.mode == "profile"
    assert result.preferred_profile == "high_quality"
    assert result.updated_at is not None


def test_set_policy_updates_existing(manager, test_video):
    """Test setting a policy updates existing one."""
    # Create initial policy
    policy1 = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="profile",
        preferred_profile="fast",
    )
    manager.set_policy(policy1)

    # Update to different mode
    policy2 = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="latest",
    )
    result = manager.set_policy(policy2)

    assert result.mode == "latest"
    assert result.preferred_profile is None

    # Verify only one policy exists
    retrieved = manager.get_policy(test_video.video_id, "transcript.segment")
    assert retrieved.mode == "latest"


def test_get_policy_retrieves_existing(manager, test_video):
    """Test getting an existing policy."""
    # Create policy
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="scene",
        mode="pinned",
        pinned_run_id="run_123",
    )
    manager.set_policy(policy)

    # Retrieve it
    retrieved = manager.get_policy(test_video.video_id, "scene")

    assert retrieved is not None
    assert retrieved.asset_id == test_video.video_id
    assert retrieved.artifact_type == "scene"
    assert retrieved.mode == "pinned"
    assert retrieved.pinned_run_id == "run_123"


def test_get_default_policy(manager):
    """Test getting default policy with empty parameters returns None."""
    default = manager.get_default_policy()

    # When called with empty strings, should return None (no default policy)
    assert default is None


def test_get_default_policy_with_params(manager, test_video):
    """Test getting default policy with asset_id and artifact_type."""
    default = manager.get_default_policy(
        asset_id=test_video.video_id, artifact_type="transcript.segment"
    )

    assert default is not None
    assert default.asset_id == test_video.video_id
    assert default.artifact_type == "transcript.segment"
    assert default.mode == "latest"
    assert default.updated_at is not None


def test_set_policy_profile_mode(manager, test_video):
    """Test setting policy with profile mode."""
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="object.detection",
        mode="profile",
        preferred_profile="balanced",
    )

    result = manager.set_policy(policy)

    assert result.mode == "profile"
    assert result.preferred_profile == "balanced"


def test_set_policy_pinned_mode(manager, test_video):
    """Test setting policy with pinned mode."""
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="face.detection",
        mode="pinned",
        pinned_run_id="run_456",
    )

    result = manager.set_policy(policy)

    assert result.mode == "pinned"
    assert result.pinned_run_id == "run_456"


def test_set_policy_latest_mode(manager, test_video):
    """Test setting policy with latest mode."""
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="place.classification",
        mode="latest",
    )

    result = manager.set_policy(policy)

    assert result.mode == "latest"


def test_set_policy_best_quality_mode(manager, test_video):
    """Test setting policy with best_quality mode."""
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="ocr.text",
        mode="best_quality",
    )

    result = manager.set_policy(policy)

    assert result.mode == "best_quality"


def test_set_policy_default_mode(manager, test_video):
    """Test setting policy with default mode."""
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="default",
    )

    result = manager.set_policy(policy)

    assert result.mode == "default"


def test_multiple_policies_different_types(manager, test_video):
    """Test setting multiple policies for different artifact types."""
    policy1 = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="profile",
        preferred_profile="high_quality",
    )
    policy2 = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="scene",
        mode="latest",
    )

    manager.set_policy(policy1)
    manager.set_policy(policy2)

    retrieved1 = manager.get_policy(test_video.video_id, "transcript.segment")
    retrieved2 = manager.get_policy(test_video.video_id, "scene")

    assert retrieved1.mode == "profile"
    assert retrieved2.mode == "latest"


def test_set_policy_with_pinned_artifact_id(manager, test_video):
    """Test setting policy with pinned artifact ID."""
    policy = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="pinned",
        pinned_run_id="run_789",
        pinned_artifact_id="artifact_abc",
    )

    result = manager.set_policy(policy)

    assert result.pinned_run_id == "run_789"
    assert result.pinned_artifact_id == "artifact_abc"


def test_set_policy_updates_timestamp(manager, test_video):
    """Test that updating a policy updates the timestamp."""
    # Create initial policy
    policy1 = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="latest",
    )
    result1 = manager.set_policy(policy1)
    first_timestamp = result1.updated_at

    # Update policy
    policy2 = SelectionPolicy(
        asset_id=test_video.video_id,
        artifact_type="transcript.segment",
        mode="profile",
        preferred_profile="fast",
    )
    result2 = manager.set_policy(policy2)
    second_timestamp = result2.updated_at

    # Timestamps should be different (second should be >= first)
    assert second_timestamp >= first_timestamp


def test_policy_validation_in_domain_model(manager, test_video):
    """Test that domain model validation is enforced."""
    # Test invalid mode
    with pytest.raises(ValueError, match="mode must be one of"):
        SelectionPolicy(
            asset_id=test_video.video_id,
            artifact_type="transcript.segment",
            mode="invalid_mode",
        )

    # Test profile mode without preferred_profile
    with pytest.raises(ValueError, match="preferred_profile required"):
        SelectionPolicy(
            asset_id=test_video.video_id,
            artifact_type="transcript.segment",
            mode="profile",
        )

    # Test pinned mode without pinned_run_id
    with pytest.raises(ValueError, match="pinned_run_id required"):
        SelectionPolicy(
            asset_id=test_video.video_id,
            artifact_type="transcript.segment",
            mode="pinned",
        )


def test_get_policy_different_assets(manager, session):
    """Test getting policies for different assets."""
    # Create two videos
    video1 = Video(
        video_id="video_1",
        file_path="/path/to/video1.mp4",
        filename="video1.mp4",
        duration=120.0,
        file_size=1024000,
        last_modified=datetime.utcnow(),
    )
    video2 = Video(
        video_id="video_2",
        file_path="/path/to/video2.mp4",
        filename="video2.mp4",
        duration=120.0,
        file_size=1024000,
        last_modified=datetime.utcnow(),
    )
    session.add(video1)
    session.add(video2)
    session.commit()

    # Set policies for both
    policy1 = SelectionPolicy(
        asset_id=video1.video_id,
        artifact_type="transcript.segment",
        mode="profile",
        preferred_profile="fast",
    )
    policy2 = SelectionPolicy(
        asset_id=video2.video_id,
        artifact_type="transcript.segment",
        mode="latest",
    )

    manager.set_policy(policy1)
    manager.set_policy(policy2)

    # Retrieve and verify
    retrieved1 = manager.get_policy(video1.video_id, "transcript.segment")
    retrieved2 = manager.get_policy(video2.video_id, "transcript.segment")

    assert retrieved1.mode == "profile"
    assert retrieved2.mode == "latest"
