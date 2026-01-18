"""Path configuration manager for video discovery."""

import uuid

from ..domain.models import PathConfig
from ..repositories.interfaces import PathConfigRepository


class PathConfigManager:
    """Manages path configurations for video discovery."""

    def __init__(self, path_config_repository: PathConfigRepository):
        self.path_config_repository = path_config_repository

    def add_path(self, path: str, recursive: bool = True) -> PathConfig:
        """Add a new path configuration."""
        # Check if path already exists
        existing = self.path_config_repository.find_by_path(path)
        if existing:
            raise ValueError(f"Path already configured: {path}")

        path_config = PathConfig(
            path_id=str(uuid.uuid4()),
            path=path,
            recursive=recursive,
        )

        return self.path_config_repository.save(path_config)

    def remove_path(self, path: str) -> bool:
        """Remove a path configuration."""
        return self.path_config_repository.delete_by_path(path)

    def list_paths(self) -> list[PathConfig]:
        """List all configured paths."""
        return self.path_config_repository.find_all()

    def get_path(self, path: str) -> PathConfig | None:
        """Get a specific path configuration."""
        return self.path_config_repository.find_by_path(path)

    def update_path(self, path: str, recursive: bool) -> PathConfig | None:
        """Update recursion setting for a path."""
        existing = self.path_config_repository.find_by_path(path)
        if not existing:
            return None

        existing.recursive = recursive
        return self.path_config_repository.save(existing)
