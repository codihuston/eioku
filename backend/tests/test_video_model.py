import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Video


def test_video_model_creation():
    """Test that Video model can be created and saved."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        Base.metadata.create_all(engine)

        session_class = sessionmaker(bind=engine)
        session = session_class()

        # Create a video record
        video = Video(
            video_id="test-video-1",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            duration=120.5,
            file_size=1024000,
            last_modified=datetime.now(),
            status="pending",
        )

        session.add(video)
        session.commit()

        # Query it back
        retrieved = session.query(Video).filter_by(video_id="test-video-1").first()
        assert retrieved is not None
        assert retrieved.filename == "video.mp4"
        assert retrieved.status == "pending"

        session.close()
