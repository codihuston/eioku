from abc import ABC, abstractmethod

from ..domain.artifacts import ArtifactEnvelope, Run, SelectionPolicy
from ..domain.models import PathConfig, Task, Video


class VideoRepository(ABC):
    """Abstract repository interface for Video persistence."""

    @abstractmethod
    def save(self, video: Video) -> Video:
        """Save video to persistence layer."""
        pass

    @abstractmethod
    def find_by_id(self, video_id: str) -> Video | None:
        """Find video by ID."""
        pass

    @abstractmethod
    def find_by_path(self, file_path: str) -> Video | None:
        """Find video by file path."""
        pass

    @abstractmethod
    def find_by_status(self, status: str) -> list[Video]:
        """Find videos by status."""
        pass

    @abstractmethod
    def delete(self, video_id: str) -> bool:
        """Delete video by ID."""
        pass

    @abstractmethod
    def find_all(self) -> list[Video]:
        """Find all videos."""
        pass


class PathConfigRepository(ABC):
    """Abstract repository interface for PathConfig persistence."""

    @abstractmethod
    def save(self, path_config: PathConfig) -> PathConfig:
        """Save path config to persistence layer."""
        pass

    @abstractmethod
    def find_all(self) -> list[PathConfig]:
        """Find all configured paths."""
        pass

    @abstractmethod
    def find_by_path(self, path: str) -> PathConfig | None:
        """Find path config by path."""
        pass

    @abstractmethod
    def delete_by_path(self, path: str) -> bool:
        """Delete path config by path."""
        pass


class TaskRepository(ABC):
    """Abstract repository interface for Task persistence."""

    @abstractmethod
    def save(self, task: Task) -> Task:
        """Save task to persistence layer."""
        pass

    @abstractmethod
    def find_by_video_id(self, video_id: str) -> list[Task]:
        """Find all tasks for a video."""
        pass

    @abstractmethod
    def find_by_status(self, status: str) -> list[Task]:
        """Find tasks by status."""
        pass

    @abstractmethod
    def find_by_task_type(self, task_type: str) -> list[Task]:
        """Find tasks by type."""
        pass

    @abstractmethod
    def find_by_video_and_type(self, video_id: str, task_type: str) -> list[Task]:
        """Find tasks by video ID and task type."""
        pass

    @abstractmethod
    def find_by_video_and_status(self, video_id: str, status: str) -> list[Task]:
        """Find tasks by video ID and status."""
        pass

    @abstractmethod
    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all tasks for a video."""
        pass


class ArtifactRepository(ABC):
    """Abstract repository interface for Artifact persistence."""

    @abstractmethod
    def create(self, artifact: ArtifactEnvelope) -> ArtifactEnvelope:
        """Create a new artifact with schema validation."""
        pass

    @abstractmethod
    def batch_create(self, artifacts: list[ArtifactEnvelope]) -> list[ArtifactEnvelope]:
        """Create multiple artifacts in a single transaction.

        Validates all artifacts before inserting any.
        Uses single transaction for all inserts.
        Rolls back entire batch on any validation error.

        Args:
            artifacts: List of artifacts to create

        Returns:
            List of created artifacts

        Raises:
            ValidationError: If any artifact fails schema validation
            DatabaseError: If database operation fails
        """
        pass

    @abstractmethod
    def get_by_id(self, artifact_id: str) -> ArtifactEnvelope | None:
        """Get artifact by ID."""
        pass

    @abstractmethod
    def get_by_asset(
        self,
        asset_id: str,
        artifact_type: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        selection: SelectionPolicy | None = None,
    ) -> list[ArtifactEnvelope]:
        """Get artifacts for an asset with optional filtering."""
        pass

    @abstractmethod
    def get_by_span(
        self,
        asset_id: str,
        artifact_type: str,
        span_start_ms: int,
        span_end_ms: int,
        selection: SelectionPolicy | None = None,
    ) -> list[ArtifactEnvelope]:
        """Get artifacts overlapping a time span."""
        pass

    @abstractmethod
    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        pass


class RunRepository(ABC):
    """Abstract repository interface for Run persistence."""

    @abstractmethod
    def create(self, run: Run) -> Run:
        """Create a new run record."""
        pass

    @abstractmethod
    def get_by_id(self, run_id: str) -> Run | None:
        """Get run by ID."""
        pass

    @abstractmethod
    def get_by_asset(self, asset_id: str) -> list[Run]:
        """Get all runs for an asset."""
        pass

    @abstractmethod
    def get_by_status(self, status: str) -> list[Run]:
        """Get all runs with a specific status."""
        pass

    @abstractmethod
    def update(self, run: Run) -> Run:
        """Update an existing run record."""
        pass

    @abstractmethod
    def delete(self, run_id: str) -> bool:
        """Delete a run record."""
        pass
