"""Configuration loader for initial path setup."""

import json
import os
from pathlib import Path

from .path_config_manager import PathConfigManager


class ConfigLoader:
    """Loads initial configuration and sets up default paths."""

    def __init__(self, path_config_manager: PathConfigManager):
        self.path_config_manager = path_config_manager

    def load_initial_config(self, config_path: str | None = None) -> None:
        """Load initial configuration from file or use defaults."""
        config = self._load_config_file(config_path)
        self._setup_initial_paths(config.get("paths", []))

    def _load_config_file(self, config_path: str | None = None) -> dict:
        """Load configuration from JSON file."""
        if config_path is None:
            # Try multiple locations in order
            possible_paths = [
                os.getenv("EIOKU_CONFIG_PATH"),
                str(Path.home() / ".eioku" / "config.json"),
                "/etc/eioku/config.json",
            ]

            for path in possible_paths:
                if path and Path(path).exists():
                    config_path = path
                    break

        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError):
                    pass

        # Return default configuration
        return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration with common video paths."""
        home_dir = Path.home()

        return {
            "paths": [
                {"path": str(home_dir / "Videos"), "recursive": True},
                {"path": "/media", "recursive": True},
                {"path": "/mnt", "recursive": True},
            ]
        }

    def _setup_initial_paths(self, path_configs: list[dict]) -> None:
        """Add new paths from config, merging with existing database paths."""
        for path_config in path_configs:
            path = path_config.get("path")
            recursive = path_config.get("recursive", True)

            if path and Path(path).exists():
                try:
                    self.path_config_manager.add_path(path, recursive)
                except ValueError:
                    # Path already exists, skip
                    pass

    def create_default_config_file(self, config_path: str | None = None) -> str:
        """Create a default configuration file."""
        if config_path is None:
            config_path = "config/eioku.json"

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        default_config = self._get_default_config()

        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2)

        return str(config_file)
