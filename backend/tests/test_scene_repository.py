import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Video
from src.domain.models import Scene as SceneDomain
from src.repositories.scene_repository import SqlSceneRepository


def test_scene_repository_crud():
    """Test SceneRepository CRUD operations."""
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
        repo = SqlSceneRepository(session)

        # Create scenes
        scene1 = SceneDomain(
            scene_id="scene-1",
            video_id="test-video-1",
            scene=1,
            start=0.0,
            end=15.5,
            thumbnail_path="/path/to/thumb1.jpg",
        )

        scene2 = SceneDomain(
            scene_id="scene-2",
            video_id="test-video-1",
            scene=2,
            start=15.5,
            end=30.0,
        )

        # Save scenes
        saved1 = repo.save(scene1)
        saved2 = repo.save(scene2)

        assert saved1.scene_id == "scene-1"
        assert saved1.has_thumbnail() is True
        assert saved2.has_thumbnail() is False

        # Find by video ID (should be ordered by scene number)
        scenes = repo.find_by_video_id("test-video-1")
        assert len(scenes) == 2
        assert scenes[0].scene == 1
        assert scenes[1].scene == 2

        # Find by scene number
        scene_1 = repo.find_by_scene_number("test-video-1", 1)
        assert scene_1 is not None
        assert scene_1.get_duration() == 15.5

        scene_3 = repo.find_by_scene_number("test-video-1", 3)
        assert scene_3 is None

        # Delete by video ID
        deleted = repo.delete_by_video_id("test-video-1")
        assert deleted is True

        remaining = repo.find_by_video_id("test-video-1")
        assert len(remaining) == 0

        session.close()
