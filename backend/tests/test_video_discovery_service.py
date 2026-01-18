"""Test VideoDiscoveryService."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.repositories.path_config_repository import SQLAlchemyPathConfigRepository
from src.repositories.video_repository import SqlVideoRepository
from src.services.path_config_manager import PathConfigManager
from src.services.video_discovery_service import VideoDiscoveryService


def test_video_discovery_service_scan():
    """Test VideoDiscoveryService discovers video files."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)
    session = session_local()

    try:
        # Set up repositories and services
        path_repo = SQLAlchemyPathConfigRepository(session)
        video_repo = SqlVideoRepository(session)
        path_manager = PathConfigManager(path_repo)
        discovery_service = VideoDiscoveryService(path_manager, video_repo)

        # Create temporary test directory with video files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)

            # Create test video files
            (test_path / "video1.mp4").write_text("fake video")
            (test_path / "video2.mov").write_text("fake video")
            (test_path / "not_video.txt").write_text("not a video")

            # Add path configuration (recursive)
            path_manager.add_path(str(test_path), recursive=True)

            # Discover videos
            discovered_videos = discovery_service.discover_videos()

            # Should find 2 video files
            assert len(discovered_videos) == 2

            # Check video properties
            video_names = {v.filename for v in discovered_videos}
            expected_names = {"video1.mp4", "video2.mov"}
            assert video_names == expected_names

    finally:
        session.close()


def test_video_discovery_service_supported_formats():
    """Test VideoDiscoveryService only processes supported formats."""
    discovery_service = VideoDiscoveryService(Mock(), Mock())

    # Test supported formats
    assert discovery_service._is_video_file(Path("video.mp4")) is True
    assert discovery_service._is_video_file(Path("video.mov")) is True
    assert discovery_service._is_video_file(Path("video.avi")) is True
    assert discovery_service._is_video_file(Path("video.mkv")) is True

    # Test case insensitive
    assert discovery_service._is_video_file(Path("video.MP4")) is True

    # Test unsupported formats
    assert discovery_service._is_video_file(Path("document.txt")) is False
