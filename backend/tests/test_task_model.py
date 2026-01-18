import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Task, Video


def test_task_model_creation():
    """Test that Task model can be created with foreign key to Video."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        Base.metadata.create_all(engine)

        session_class = sessionmaker(bind=engine)
        session = session_class()

        # Create a video first
        video = Video(
            video_id="test-video-1",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=datetime.now(),
            status="pending"
        )
        session.add(video)
        session.commit()

        # Create processing task
        task = Task(
            task_id="task-1",
            video_id="test-video-1",
            task_type="transcription",
            status="pending",
            priority=1,
            dependencies=["task-0"]
        )

        session.add(task)
        session.commit()

        # Query it back
        retrieved = session.query(Task).filter_by(task_id="task-1").first()
        assert retrieved is not None
        assert retrieved.task_type == "transcription"
        assert retrieved.status == "pending"
        assert retrieved.priority == 1
        assert retrieved.dependencies == ["task-0"]

        session.close()
