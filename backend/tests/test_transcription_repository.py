import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Video
from src.domain.models import Transcription as TranscriptionDomain
from src.repositories.transcription_repository import SqlTranscriptionRepository


def test_transcription_repository_crud():
    """Test TranscriptionRepository CRUD operations."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        Base.metadata.create_all(engine)

        session_class = sessionmaker(bind=engine)
        session = session_class()

        # Create video first
        video = Video(
            video_id="test-video-1",
            file_path="/test/video.mp4",
            filename="video.mp4",
            last_modified=datetime.now(),
            status="pending",
            file_hash=None,
        )
        session.add(video)
        session.commit()

        # Test repository
        repo = SqlTranscriptionRepository(session)

        # Create transcription
        transcription = TranscriptionDomain(
            segment_id="seg-1",
            video_id="test-video-1",
            text="Hello world",
            start=0.0,
            end=2.5,
            confidence=0.95,
            speaker="speaker_1",
        )

        saved = repo.save(transcription)
        assert saved.segment_id == "seg-1"
        assert saved.text == "Hello world"

        # Find by video ID
        transcriptions = repo.find_by_video_id("test-video-1")
        assert len(transcriptions) == 1
        assert transcriptions[0].text == "Hello world"

        # Find by time range
        in_range = repo.find_by_time_range("test-video-1", 0.0, 3.0)
        assert len(in_range) == 1

        out_of_range = repo.find_by_time_range("test-video-1", 3.0, 5.0)
        assert len(out_of_range) == 0

        # Delete by video ID
        deleted = repo.delete_by_video_id("test-video-1")
        assert deleted is True

        remaining = repo.find_by_video_id("test-video-1")
        assert len(remaining) == 0

        session.close()
