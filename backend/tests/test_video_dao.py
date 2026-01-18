import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.dao import VideoDAO
from src.database.models import Video


def test_video_dao_crud():
    """Test Video DAO CRUD operations."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        Base.metadata.create_all(engine)

        session_class = sessionmaker(bind=engine)
        session = session_class()
        dao = VideoDAO(session)

        # Create
        video = Video(
            video_id="test-1",
            file_path="/test/video.mp4",
            filename="video.mp4",
            last_modified=datetime.now(),
            status="pending",
        )
        created = dao.create(video)
        assert created.video_id == "test-1"

        # Read
        retrieved = dao.get_by_id("test-1")
        assert retrieved is not None
        assert retrieved.filename == "video.mp4"

        # Update
        retrieved.status = "completed"
        updated = dao.update(retrieved)
        assert updated.status == "completed"

        # Delete
        deleted = dao.delete("test-1")
        assert deleted is True
        assert dao.get_by_id("test-1") is None

        session.close()
