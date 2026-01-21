"""Tests for RunRepository implementation."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database.models import Video as VideoEntity
from src.domain.artifacts import Run
from src.repositories.run_repository import SqlRunRepository


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


@pytest.fixture
def repository(session):
    """Create run repository instance."""
    return SqlRunRepository(session)


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


@pytest.fixture
def sample_run(test_video):
    """Create a sample run."""
    return Run(
        run_id="run_1",
        asset_id=test_video.video_id,
        pipeline_profile="balanced",
        started_at=datetime.now(),
        status="running",
    )


def test_create_run(repository, sample_run):
    """Test creating a run."""
    result = repository.create(sample_run)

    assert result.run_id == sample_run.run_id
    assert result.asset_id == sample_run.asset_id
    assert result.pipeline_profile == sample_run.pipeline_profile
    assert result.status == sample_run.status
    assert result.started_at == sample_run.started_at
    assert result.finished_at is None
    assert result.error is None


def test_get_run_by_id(repository, sample_run):
    """Test getting a run by ID."""
    repository.create(sample_run)

    result = repository.get_by_id(sample_run.run_id)

    assert result is not None
    assert result.run_id == sample_run.run_id
    assert result.asset_id == sample_run.asset_id
    assert result.status == sample_run.status


def test_get_run_by_id_not_found(repository):
    """Test getting a non-existent run returns None."""
    result = repository.get_by_id("nonexistent_run")
    assert result is None


def test_get_runs_by_asset(repository, test_video):
    """Test getting all runs for an asset."""
    run1 = Run(
        run_id="run_1",
        asset_id=test_video.video_id,
        pipeline_profile="fast",
        started_at=datetime.now(),
        status="completed",
    )
    run2 = Run(
        run_id="run_2",
        asset_id=test_video.video_id,
        pipeline_profile="balanced",
        started_at=datetime.now(),
        status="running",
    )

    repository.create(run1)
    repository.create(run2)

    results = repository.get_by_asset(test_video.video_id)

    assert len(results) == 2
    assert any(r.run_id == "run_1" for r in results)
    assert any(r.run_id == "run_2" for r in results)


def test_get_runs_by_status(repository, test_video):
    """Test getting runs by status."""
    run1 = Run(
        run_id="run_1",
        asset_id=test_video.video_id,
        pipeline_profile="fast",
        started_at=datetime.now(),
        status="completed",
    )
    run2 = Run(
        run_id="run_2",
        asset_id=test_video.video_id,
        pipeline_profile="balanced",
        started_at=datetime.now(),
        status="running",
    )
    run3 = Run(
        run_id="run_3",
        asset_id=test_video.video_id,
        pipeline_profile="high_quality",
        started_at=datetime.now(),
        status="completed",
    )

    repository.create(run1)
    repository.create(run2)
    repository.create(run3)

    completed_runs = repository.get_by_status("completed")
    running_runs = repository.get_by_status("running")

    assert len(completed_runs) == 2
    assert len(running_runs) == 1
    assert running_runs[0].run_id == "run_2"


def test_update_run(repository, sample_run):
    """Test updating a run."""
    repository.create(sample_run)

    # Complete the run
    finished_at = datetime.now()
    sample_run.complete(finished_at)

    result = repository.update(sample_run)

    assert result.status == "completed"
    assert result.finished_at == finished_at
    assert result.error is None


def test_update_run_with_failure(repository, sample_run):
    """Test updating a run with failure."""
    repository.create(sample_run)

    # Fail the run
    finished_at = datetime.now()
    error_message = "Processing failed due to timeout"
    sample_run.fail(error_message, finished_at)

    result = repository.update(sample_run)

    assert result.status == "failed"
    assert result.finished_at == finished_at
    assert result.error == error_message


def test_update_nonexistent_run(repository):
    """Test updating a non-existent run raises error."""
    run = Run(
        run_id="nonexistent_run",
        asset_id="test_video_1",
        pipeline_profile="balanced",
        started_at=datetime.now(),
        status="running",
    )

    with pytest.raises(ValueError, match="Run not found"):
        repository.update(run)


def test_delete_run(repository, sample_run):
    """Test deleting a run."""
    repository.create(sample_run)

    result = repository.delete(sample_run.run_id)

    assert result is True
    assert repository.get_by_id(sample_run.run_id) is None


def test_delete_nonexistent_run(repository):
    """Test deleting a non-existent run returns False."""
    result = repository.delete("nonexistent_run")
    assert result is False


def test_run_lifecycle(repository, sample_run):
    """Test complete run lifecycle: create -> update -> complete."""
    # Create running run
    created = repository.create(sample_run)
    assert created.status == "running"
    assert created.finished_at is None

    # Complete the run
    finished_at = datetime.now()
    sample_run.complete(finished_at)
    updated = repository.update(sample_run)

    assert updated.status == "completed"
    assert updated.finished_at == finished_at
    assert updated.error is None

    # Verify persistence
    retrieved = repository.get_by_id(sample_run.run_id)
    assert retrieved.status == "completed"
    assert retrieved.finished_at == finished_at
