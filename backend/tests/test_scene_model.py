import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Scene, Video


def test_scene_model_creation():
    """Test that Scene model can be created with foreign key to Video."""
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

        # Create scene
        scene = Scene(
            scene_id="scene-1",
            video_id="test-video-1",
            scene=1,
            start=0.0,
            end=15.5,
            thumbnail_path="/path/to/thumbnail.jpg",
        )

        session.add(scene)
        session.commit()

        # Query it back
        retrieved = session.query(Scene).filter_by(scene_id="scene-1").first()
        assert retrieved is not None
        assert retrieved.video_id == "test-video-1"
        assert retrieved.scene == 1
        assert retrieved.start == 0.0
        assert retrieved.end == 15.5

        session.close()
