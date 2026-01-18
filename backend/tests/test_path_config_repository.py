"""Test PathConfigRepository implementation."""

import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.domain.models import PathConfig
from src.repositories.path_config_repository import SQLAlchemyPathConfigRepository


def test_path_config_repository_crud():
    """Test PathConfig repository CRUD operations."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)
    session = session_local()

    try:
        repo = SQLAlchemyPathConfigRepository(session)

        # Create test path config
        path_config = PathConfig(
            path_id="path_1",
            path="/home/user/videos",
            recursive=True,
        )

        # Test save
        saved_config = repo.save(path_config)
        assert saved_config.path_id == "path_1"
        assert saved_config.path == "/home/user/videos"
        assert saved_config.recursive is True
        assert saved_config.added_at is not None

        # Test find_all
        all_configs = repo.find_all()
        assert len(all_configs) == 1
        assert all_configs[0].path_id == "path_1"

        # Test find_by_path
        found_config = repo.find_by_path("/home/user/videos")
        assert found_config is not None
        assert found_config.path_id == "path_1"

        # Test find_by_path with non-existent path
        not_found = repo.find_by_path("/nonexistent/path")
        assert not_found is None

        # Add more path configs
        path_config2 = PathConfig(
            path_id="path_2",
            path="/media/external/videos",
            recursive=False,
        )

        path_config3 = PathConfig(
            path_id="path_3",
            path="/shared/content",
            recursive=True,
        )

        repo.save(path_config2)
        repo.save(path_config3)

        # Test multiple configs (should be ordered by added_at desc)
        all_configs = repo.find_all()
        assert len(all_configs) == 3
        # Most recently added should be first
        assert all_configs[0].path_id == "path_3"

        # Test different recursive settings
        non_recursive = repo.find_by_path("/media/external/videos")
        assert non_recursive.recursive is False

        # Test delete_by_path
        deleted = repo.delete_by_path("/home/user/videos")
        assert deleted is True

        # Verify deletion
        configs_after_delete = repo.find_all()
        assert len(configs_after_delete) == 2

        deleted_config = repo.find_by_path("/home/user/videos")
        assert deleted_config is None

        # Test delete non-existent path
        deleted_none = repo.delete_by_path("/nonexistent/path")
        assert deleted_none is False

    finally:
        session.close()


def test_path_config_domain_methods():
    """Test PathConfig domain model methods."""
    # Test recursive path
    recursive_config = PathConfig(
        path_id="path_1",
        path="/home/user/videos",
        recursive=True,
    )

    assert recursive_config.is_recursive() is True
    assert recursive_config.added_at is not None

    # Test non-recursive path
    non_recursive_config = PathConfig(
        path_id="path_2",
        path="/media/external",
        recursive=False,
    )

    assert non_recursive_config.is_recursive() is False

    # Test with explicit added_at
    specific_time = datetime(2024, 1, 15, 10, 30, 0)
    timed_config = PathConfig(
        path_id="path_3",
        path="/custom/path",
        recursive=True,
        added_at=specific_time,
    )

    assert timed_config.added_at == specific_time
    assert timed_config.is_recursive() is True
