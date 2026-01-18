"""Test ConfigLoader service."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.services.config_loader import ConfigLoader


def test_config_loader_with_existing_config():
    """Test ConfigLoader with existing configuration file."""
    mock_manager = Mock()
    mock_manager.list_paths.return_value = []  # No existing paths

    loader = ConfigLoader(mock_manager)

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "paths": [
                {"path": "/test/videos", "recursive": True},
                {"path": "/test/media", "recursive": False},
            ]
        }
        json.dump(config, f)
        config_path = f.name

    try:
        with patch("pathlib.Path.exists", return_value=True):
            loader.load_initial_config(config_path)

        # Should attempt to add both paths
        assert mock_manager.add_path.call_count == 2
        mock_manager.add_path.assert_any_call("/test/videos", True)
        mock_manager.add_path.assert_any_call("/test/media", False)

    finally:
        Path(config_path).unlink()


def test_config_loader_with_default_config():
    """Test ConfigLoader with default configuration."""
    mock_manager = Mock()
    mock_manager.list_paths.return_value = []  # No existing paths

    loader = ConfigLoader(mock_manager)

    with patch("pathlib.Path.exists", return_value=True):
        loader.load_initial_config("/nonexistent/config.json")

        # Should attempt to add default paths
        assert mock_manager.add_path.call_count == 3


def test_config_loader_merge_behavior():
    """Test ConfigLoader properly merges new paths with existing ones."""
    mock_manager = Mock()

    # Mock existing path
    existing_path = Mock()
    existing_path.path = "/existing/path"
    mock_manager.list_paths.return_value = [existing_path]

    loader = ConfigLoader(mock_manager)

    # Create config with mix of new and existing paths
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "paths": [
                {"path": "/existing/path", "recursive": True},  # Should be skipped
                {"path": "/new/path1", "recursive": True},  # Should be added
                {"path": "/new/path2", "recursive": False},  # Should be added
            ]
        }
        json.dump(config, f)
        config_path = f.name

    try:
        with patch("pathlib.Path.exists", return_value=True):
            loader.load_initial_config(config_path)

        # Should attempt to add all paths (duplicates handled by manager)
        assert mock_manager.add_path.call_count == 3
        mock_manager.add_path.assert_any_call("/existing/path", True)
        mock_manager.add_path.assert_any_call("/new/path1", True)
        mock_manager.add_path.assert_any_call("/new/path2", False)

    finally:
        Path(config_path).unlink()


def test_config_loader_merges_with_existing_paths():
    """Test ConfigLoader merges config paths with existing paths."""
    mock_manager = Mock()
    mock_manager.list_paths.return_value = [Mock()]  # Existing paths

    loader = ConfigLoader(mock_manager)

    with patch("pathlib.Path.exists", return_value=True):
        loader.load_initial_config()

        # Should attempt to add new paths from config (merge behavior)
        assert mock_manager.add_path.call_count >= 1


def test_config_loader_handles_invalid_config():
    """Test ConfigLoader handles invalid configuration gracefully."""
    mock_manager = Mock()
    mock_manager.list_paths.return_value = []

    loader = ConfigLoader(mock_manager)

    # Create invalid JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json content")
        config_path = f.name

    try:
        with patch("pathlib.Path.exists", return_value=True):
            loader.load_initial_config(config_path)

            # Should fall back to default config and add paths
            assert mock_manager.add_path.call_count == 3

    finally:
        Path(config_path).unlink()


def test_create_default_config_file():
    """Test creating default configuration file."""
    mock_manager = Mock()
    loader = ConfigLoader(mock_manager)

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"

        created_path = loader.create_default_config_file(str(config_path))

        assert Path(created_path).exists()

        # Verify content
        with open(created_path) as f:
            config = json.load(f)

        assert "paths" in config
        assert len(config["paths"]) == 3
        assert any(p["path"] == str(Path.home() / "Videos") for p in config["paths"])
