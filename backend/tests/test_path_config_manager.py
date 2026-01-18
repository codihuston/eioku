"""Test PathConfigManager service."""

import tempfile
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.domain.models import PathConfig
from src.repositories.path_config_repository import SQLAlchemyPathConfigRepository
from src.services.path_config_manager import PathConfigManager


def test_path_config_manager_operations():
    """Test PathConfigManager CRUD operations."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)
    session = session_local()

    try:
        repo = SQLAlchemyPathConfigRepository(session)
        manager = PathConfigManager(repo)

        # Test add_path
        path_config = manager.add_path("/home/user/videos", recursive=True)
        assert path_config.path == "/home/user/videos"
        assert path_config.recursive is True
        assert path_config.path_id is not None

        # Test add duplicate path (should raise error)
        with pytest.raises(ValueError, match="Path already configured"):
            manager.add_path("/home/user/videos")

        # Test list_paths
        paths = manager.list_paths()
        assert len(paths) == 1
        assert paths[0].path == "/home/user/videos"

        # Test get_path
        found_path = manager.get_path("/home/user/videos")
        assert found_path is not None
        assert found_path.path == "/home/user/videos"

        # Test get non-existent path
        not_found = manager.get_path("/nonexistent")
        assert not_found is None

        # Test add more paths
        manager.add_path("/media/external", recursive=False)
        manager.add_path("/shared/content", recursive=True)

        all_paths = manager.list_paths()
        assert len(all_paths) == 3

        # Test update_path
        updated = manager.update_path("/media/external", recursive=True)
        assert updated is not None
        assert updated.recursive is True

        # Test update non-existent path
        not_updated = manager.update_path("/nonexistent", recursive=True)
        assert not_updated is None

        # Test remove_path
        removed = manager.remove_path("/home/user/videos")
        assert removed is True

        # Verify removal
        paths_after_remove = manager.list_paths()
        assert len(paths_after_remove) == 2

        # Test remove non-existent path
        not_removed = manager.remove_path("/nonexistent")
        assert not_removed is False

    finally:
        session.close()


def test_path_config_manager_with_mock():
    """Test PathConfigManager with mocked repository."""
    mock_repo = Mock()
    manager = PathConfigManager(mock_repo)

    # Test add_path with no existing path
    mock_repo.find_by_path.return_value = None
    mock_path_config = PathConfig("id1", "/test/path", True)
    mock_repo.save.return_value = mock_path_config

    result = manager.add_path("/test/path", True)

    mock_repo.find_by_path.assert_called_once_with("/test/path")
    mock_repo.save.assert_called_once()
    assert result == mock_path_config

    # Test add_path with existing path
    mock_repo.reset_mock()
    mock_repo.find_by_path.return_value = mock_path_config

    with pytest.raises(ValueError):
        manager.add_path("/test/path")

    # Test list_paths
    mock_repo.reset_mock()
    mock_paths = [mock_path_config]
    mock_repo.find_all.return_value = mock_paths

    result = manager.list_paths()

    mock_repo.find_all.assert_called_once()
    assert result == mock_paths

    # Test remove_path
    mock_repo.reset_mock()
    mock_repo.delete_by_path.return_value = True

    result = manager.remove_path("/test/path")

    mock_repo.delete_by_path.assert_called_once_with("/test/path")
    assert result is True
