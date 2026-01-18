import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Face, Video


def test_face_model_creation():
    """Test that Face model can be created with JSON fields."""
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

        # Create face detection result
        face = Face(
            face_id="face-1",
            video_id="test-video-1",
            person_id="person_123",
            timestamps=[2.1, 4.5, 7.8],
            bounding_boxes=[
                {"x": 150, "y": 80, "width": 120, "height": 150},
                {"x": 155, "y": 85, "width": 115, "height": 145},
                {"x": 148, "y": 82, "width": 118, "height": 148},
            ],
            confidence=0.92,
        )

        session.add(face)
        session.commit()

        # Query it back
        retrieved = session.query(Face).filter_by(face_id="face-1").first()
        assert retrieved is not None
        assert retrieved.person_id == "person_123"
        assert retrieved.confidence == 0.92
        assert len(retrieved.timestamps) == 3

        session.close()
