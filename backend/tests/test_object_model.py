import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Object, Video


def test_object_model_creation():
    """Test that Object model can be created with JSON fields."""
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

        # Create object detection result
        obj = Object(
            object_id="obj-1",
            video_id="test-video-1",
            label="person",
            timestamps=[1.5, 3.2, 5.8],
            bounding_boxes=[
                {"x": 100, "y": 50, "width": 200, "height": 300},
                {"x": 110, "y": 55, "width": 190, "height": 295},
                {"x": 105, "y": 52, "width": 195, "height": 298},
            ],
        )

        session.add(obj)
        session.commit()

        # Query it back
        retrieved = session.query(Object).filter_by(object_id="obj-1").first()
        assert retrieved is not None
        assert retrieved.label == "person"
        assert retrieved.timestamps == [1.5, 3.2, 5.8]
        assert len(retrieved.bounding_boxes) == 3

        session.close()
