from abc import ABC, abstractmethod

from ..domain.artifacts import ArtifactEnvelope, SelectionPolicy
from ..domain.models import (
    Face,
    Object,
    PathConfig,
    Scene,
    Task,
    Topic,
    Transcription,
    Video,
)


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


class TranscriptionRepository(ABC):
    """Abstract repository interface for Transcription persistence."""

    @abstractmethod
    def save(self, transcription: Transcription) -> Transcription:
        """Save transcription to persistence layer."""
        pass

    @abstractmethod
    def find_by_video_id(self, video_id: str) -> list[Transcription]:
        """Find all transcriptions for a video."""
        pass

    @abstractmethod
    def find_by_time_range(
        self, video_id: str, start: float, end: float
    ) -> list[Transcription]:
        """Find transcriptions within time range."""
        pass

    @abstractmethod
    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all transcriptions for a video."""
        pass


class SceneRepository(ABC):
    """Abstract repository interface for Scene persistence."""

    @abstractmethod
    def save(self, scene: Scene) -> Scene:
        """Save scene to persistence layer."""
        pass

    @abstractmethod
    def find_by_video_id(self, video_id: str) -> list[Scene]:
        """Find all scenes for a video."""
        pass

    @abstractmethod
    def find_by_scene_number(self, video_id: str, scene_number: int) -> Scene | None:
        """Find scene by number within a video."""
        pass

    @abstractmethod
    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all scenes for a video."""
        pass


class ObjectRepository(ABC):
    """Abstract repository interface for Object persistence."""

    @abstractmethod
    def save(self, obj: Object) -> Object:
        """Save object to persistence layer."""
        pass

    @abstractmethod
    def find_by_video_id(self, video_id: str) -> list[Object]:
        """Find all objects for a video."""
        pass

    @abstractmethod
    def find_by_label(self, video_id: str, label: str) -> list[Object]:
        """Find objects by label within a video."""
        pass

    @abstractmethod
    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all objects for a video."""
        pass


class FaceRepository(ABC):
    """Abstract repository interface for Face persistence."""

    @abstractmethod
    def save(self, face: Face) -> Face:
        """Save face to persistence layer."""
        pass

    @abstractmethod
    def find_by_video_id(self, video_id: str) -> list[Face]:
        """Find all faces for a video."""
        pass

    @abstractmethod
    def find_by_person_id(self, video_id: str, person_id: str) -> list[Face]:
        """Find faces by person ID within a video."""
        pass

    @abstractmethod
    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all faces for a video."""
        pass


class TopicRepository(ABC):
    """Abstract repository interface for Topic persistence."""

    @abstractmethod
    def save(self, topic: Topic) -> Topic:
        """Save topic to persistence layer."""
        pass

    @abstractmethod
    def find_by_video_id(self, video_id: str) -> list[Topic]:
        """Find all topics for a video."""
        pass

    @abstractmethod
    def find_by_label(self, video_id: str, label: str) -> list[Topic]:
        """Find topics by label within a video."""
        pass

    @abstractmethod
    def get_aggregated_topics(self) -> list[dict]:
        """Get aggregated topics across all videos."""
        pass

    @abstractmethod
    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all topics for a video."""
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
