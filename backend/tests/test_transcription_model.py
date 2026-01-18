import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Transcription, Video


def test_transcription_model_creation():
    """Test that Transcription model can be created with foreign key to Video."""
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
            status="pending",
        )
        session.add(video)
        session.commit()

        # Create transcription segments
        transcription = Transcription(
            segment_id="seg-1",
            video_id="test-video-1",
            text="Hello world, this is a test transcription.",
            start=0.0,
            end=5.2,
            confidence=0.95,
            speaker="speaker_1",
        )

        session.add(transcription)
        session.commit()

        # Query it back
        retrieved = session.query(Transcription).filter_by(segment_id="seg-1").first()
        assert retrieved is not None
        assert retrieved.text == "Hello world, this is a test transcription."
        assert retrieved.video_id == "test-video-1"
        assert retrieved.confidence == 0.95

        session.close()
