"""Test TaskRepository implementation."""

import tempfile
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Video
from src.domain.models import Task
from src.repositories.task_repository import SQLAlchemyTaskRepository


@pytest.fixture(scope="module")
def test_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)

    yield session_local


@pytest.fixture
def session(test_db):
    """Create a new session for each test."""
    session = test_db()
    yield session
    session.rollback()  # Rollback any changes
    session.close()


def test_task_repository_crud(session):
    """Test Task repository CRUD operations."""
    # Create test video first
    video = Video(
        video_id="video_1",
        file_path="/test/video.mp4",
        filename="video.mp4",
        last_modified=datetime.utcnow(),
        status="pending",
    )
    session.add(video)
    session.commit()

    repo = SQLAlchemyTaskRepository(session)

    # Create test task
    task = Task(
        task_id="task_1",
        video_id="video_1",
        task_type="transcription",
        status="pending",
        priority=2,
        dependencies=["task_0"],
    )

    # Test save
    saved_task = repo.save(task)
    assert saved_task.task_id == "task_1"
    assert saved_task.video_id == "video_1"
    assert saved_task.task_type == "transcription"
    assert saved_task.status == "pending"
    assert saved_task.priority == 2
    assert saved_task.dependencies == ["task_0"]
    assert saved_task.created_at is not None

    # Test find_by_video_id
    video_tasks = repo.find_by_video_id("video_1")
    assert len(video_tasks) == 1
    assert video_tasks[0].task_id == "task_1"

    # Test find_by_status
    pending_tasks = repo.find_by_status("pending")
    assert len(pending_tasks) == 1
    assert pending_tasks[0].status == "pending"

    # Test find_by_task_type
    transcription_tasks = repo.find_by_task_type("transcription")
    assert len(transcription_tasks) == 1
    assert transcription_tasks[0].task_type == "transcription"

    # Add more tasks for testing
    task2 = Task(
        task_id="task_2",
        video_id="video_1",
        task_type="scene_detection",
        status="running",
        priority=3,
    )

    task3 = Task(
        task_id="task_3",
        video_id="video_1",
        task_type="object_detection",
        status="completed",
        priority=1,
    )

    repo.save(task2)
    repo.save(task3)

    # Test multiple tasks (should be ordered by priority desc, then created_at asc)
    all_video_tasks = repo.find_by_video_id("video_1")
    assert len(all_video_tasks) == 3
    assert all_video_tasks[0].priority >= all_video_tasks[1].priority

    # Test status filtering
    running_tasks = repo.find_by_status("running")
    assert len(running_tasks) == 1
    assert running_tasks[0].task_id == "task_2"

    completed_tasks = repo.find_by_status("completed")
    assert len(completed_tasks) == 1
    assert completed_tasks[0].task_id == "task_3"

    # Test task type filtering
    scene_tasks = repo.find_by_task_type("scene_detection")
    assert len(scene_tasks) == 1
    assert scene_tasks[0].task_type == "scene_detection"

    # Test delete_by_video_id
    deleted = repo.delete_by_video_id("video_1")
    assert deleted is True

    # Verify deletion
    tasks_after_delete = repo.find_by_video_id("video_1")
    assert len(tasks_after_delete) == 0

    # Test delete non-existent video
    deleted_none = repo.delete_by_video_id("nonexistent")
    assert deleted_none is False


def test_task_domain_methods():
    """Test Task domain model methods."""
    task = Task(
        task_id="task_1",
        video_id="video_1",
        task_type="transcription",
        status="pending",
        priority=2,
        dependencies=["task_0"],
    )

    # Test status checks
    assert task.is_pending() is True
    assert task.is_running() is False
    assert task.is_completed() is False
    assert task.is_failed() is False

    # Test start
    task.start()
    assert task.is_pending() is False
    assert task.is_running() is True
    assert task.started_at is not None

    # Test complete
    task.complete()
    assert task.is_running() is False
    assert task.is_completed() is True
    assert task.completed_at is not None

    # Test failed task
    failed_task = Task(
        task_id="task_2",
        video_id="video_1",
        task_type="object_detection",
    )

    failed_task.fail("Processing error occurred")
    assert failed_task.is_failed() is True
    assert failed_task.error == "Processing error occurred"
    assert failed_task.completed_at is not None

    # Test task with no dependencies
    simple_task = Task(
        task_id="task_3",
        video_id="video_1",
        task_type="scene_detection",
    )

    assert simple_task.dependencies == []
    assert simple_task.priority == 1  # default
    assert simple_task.status == "pending"  # default


def test_atomic_dequeue_pending_task(session):
    """Test atomic task dequeue with SELECT FOR UPDATE."""
    # Create test video
    video = Video(
        video_id="video_atomic_test",
        file_path="/test/atomic_video.mp4",
        filename="atomic_video.mp4",
        last_modified=datetime.utcnow(),
        status="pending",
    )
    session.add(video)
    session.commit()

    repo = SQLAlchemyTaskRepository(session)

    # Create multiple pending tasks with different priorities
    task1 = Task(
        task_id="atomic_task_1",
        video_id="video_atomic_test",
        task_type="transcription",
        status="pending",
        priority=1,
    )
    task2 = Task(
        task_id="atomic_task_2",
        video_id="video_atomic_test",
        task_type="transcription",
        status="pending",
        priority=3,  # Higher priority
    )
    task3 = Task(
        task_id="atomic_task_3",
        video_id="video_atomic_test",
        task_type="transcription",
        status="running",  # Already running
        priority=5,
    )

    repo.save(task1)
    repo.save(task2)
    repo.save(task3)

    # Dequeue should get highest priority pending task
    dequeued = repo.atomic_dequeue_pending_task("transcription")
    assert dequeued is not None
    assert dequeued.task_id == "atomic_task_2"  # Highest priority pending
    assert dequeued.status == "running"  # Should be marked as running
    assert dequeued.started_at is not None

    # Dequeue again should get the next pending task
    dequeued2 = repo.atomic_dequeue_pending_task("transcription")
    assert dequeued2 is not None
    assert dequeued2.task_id == "atomic_task_1"
    assert dequeued2.status == "running"

    # No more pending tasks of this type
    dequeued3 = repo.atomic_dequeue_pending_task("transcription")
    assert dequeued3 is None

    # Different task type should return None
    dequeued4 = repo.atomic_dequeue_pending_task("scene_detection")
    assert dequeued4 is None
