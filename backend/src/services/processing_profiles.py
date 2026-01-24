"""Processing profile configuration for video processing."""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .task_orchestration import TaskType
from .worker_pool_manager import ResourceType, WorkerConfig

logger = logging.getLogger(__name__)


class ProfileType(Enum):
    """Available processing profiles."""

    BALANCED = "balanced"
    SEARCH_FIRST = "search_first"
    VISUAL_FIRST = "visual_first"
    LOW_RESOURCE = "low_resource"


@dataclass
class TaskSettings:
    """Task-specific settings for processing."""

    max_concurrent_videos: int = 5
    frame_sampling_interval: int = 30
    face_sampling_interval_seconds: float = 5.0
    transcription_model: str = "large-v3"
    object_detection_model: str = "yolov8n.pt"
    face_detection_model: str = "yolov8n-face.pt"


@dataclass
class ProcessingProfile:
    """Complete processing profile configuration."""

    name: str
    description: str
    worker_configs: dict[str, WorkerConfig]
    task_settings: TaskSettings

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "workers": {
                task_type: {
                    "count": config.worker_count,
                    "priority": config.priority,
                    "resource": config.resource_type.value,
                }
                for task_type, config in self.worker_configs.items()
            },
            "task_settings": asdict(self.task_settings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingProfile":
        """Create profile from dictionary."""
        worker_configs = {}
        for task_type_str, worker_data in data.get("workers", {}).items():
            task_type = TaskType(task_type_str)
            worker_configs[task_type_str] = WorkerConfig(
                task_type=task_type,
                worker_count=worker_data["count"],
                resource_type=ResourceType(worker_data["resource"]),
                priority=worker_data["priority"],
            )

        task_settings_data = data.get("task_settings", {})
        task_settings = TaskSettings(**task_settings_data)

        return cls(
            name=data["name"],
            description=data["description"],
            worker_configs=worker_configs,
            task_settings=task_settings,
        )


class ProfileManager:
    """Manages processing profiles and configurations."""

    def __init__(self, config_dir: str | None = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config_dir.mkdir(exist_ok=True)
        self.profiles: dict[str, ProcessingProfile] = {}
        self._load_default_profiles()

    def get_profile(self, profile_name: str) -> ProcessingProfile:
        """Get a processing profile by name."""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        return self.profiles[profile_name]

    def list_profiles(self) -> dict[str, str]:
        """List available profiles with descriptions."""
        return {name: profile.description for name, profile in self.profiles.items()}

    def add_profile(self, profile: ProcessingProfile) -> None:
        """Add a custom processing profile."""
        self.profiles[profile.name] = profile
        logger.info(f"Added processing profile: {profile.name}")

    def save_profile(self, profile_name: str, file_path: str | None = None) -> None:
        """Save a profile to file."""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")

        profile = self.profiles[profile_name]

        if file_path is None:
            file_path = self.config_dir / f"{profile_name}.json"
        else:
            file_path = Path(file_path)

        with open(file_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

        logger.info(f"Saved profile '{profile_name}' to {file_path}")

    def load_profile(self, file_path: str) -> ProcessingProfile:
        """Load a profile from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Profile file not found: {file_path}")

        with open(file_path) as f:
            data = json.load(f)

        profile = ProcessingProfile.from_dict(data)
        self.profiles[profile.name] = profile

        logger.info(f"Loaded profile '{profile.name}' from {file_path}")
        return profile

    def _load_default_profiles(self) -> None:
        """Load default processing profiles."""
        # Balanced Profile
        balanced = ProcessingProfile(
            name="balanced",
            description="Even resource distribution, optimized for general use",
            worker_configs={
                TaskType.HASH.value: WorkerConfig(
                    TaskType.HASH, 4, ResourceType.CPU, 1
                ),
                TaskType.TRANSCRIPTION.value: WorkerConfig(
                    TaskType.TRANSCRIPTION, 2, ResourceType.CPU, 2
                ),
                TaskType.SCENE_DETECTION.value: WorkerConfig(
                    TaskType.SCENE_DETECTION, 2, ResourceType.CPU, 3
                ),
                TaskType.OBJECT_DETECTION.value: WorkerConfig(
                    TaskType.OBJECT_DETECTION, 2, ResourceType.GPU, 3
                ),
                TaskType.FACE_DETECTION.value: WorkerConfig(
                    TaskType.FACE_DETECTION, 2, ResourceType.GPU, 3
                ),
                TaskType.OCR.value: WorkerConfig(TaskType.OCR, 2, ResourceType.GPU, 3),
                TaskType.PLACE_DETECTION.value: WorkerConfig(
                    TaskType.PLACE_DETECTION, 2, ResourceType.GPU, 3
                ),
                TaskType.TOPIC_EXTRACTION.value: WorkerConfig(
                    TaskType.TOPIC_EXTRACTION, 1, ResourceType.CPU, 4
                ),
                TaskType.EMBEDDING_GENERATION.value: WorkerConfig(
                    TaskType.EMBEDDING_GENERATION, 2, ResourceType.CPU, 2
                ),
                TaskType.THUMBNAIL_GENERATION.value: WorkerConfig(
                    TaskType.THUMBNAIL_GENERATION, 1, ResourceType.CPU, 4
                ),
            },
            task_settings=TaskSettings(
                max_concurrent_videos=5,
                frame_sampling_interval=30,
                face_sampling_interval_seconds=5.0,
            ),
        )

        # Search First Profile
        search_first = ProcessingProfile(
            name="search_first",
            description="Prioritize getting videos searchable quickly",
            worker_configs={
                TaskType.HASH.value: WorkerConfig(
                    TaskType.HASH, 6, ResourceType.CPU, 1
                ),
                TaskType.TRANSCRIPTION.value: WorkerConfig(
                    TaskType.TRANSCRIPTION, 4, ResourceType.CPU, 1
                ),
                TaskType.SCENE_DETECTION.value: WorkerConfig(
                    TaskType.SCENE_DETECTION, 1, ResourceType.CPU, 4
                ),
                TaskType.OBJECT_DETECTION.value: WorkerConfig(
                    TaskType.OBJECT_DETECTION, 1, ResourceType.GPU, 4
                ),
                TaskType.FACE_DETECTION.value: WorkerConfig(
                    TaskType.FACE_DETECTION, 1, ResourceType.GPU, 4
                ),
                TaskType.OCR.value: WorkerConfig(TaskType.OCR, 2, ResourceType.GPU, 2),
                TaskType.PLACE_DETECTION.value: WorkerConfig(
                    TaskType.PLACE_DETECTION, 1, ResourceType.GPU, 4
                ),
                TaskType.TOPIC_EXTRACTION.value: WorkerConfig(
                    TaskType.TOPIC_EXTRACTION, 1, ResourceType.CPU, 3
                ),
                TaskType.EMBEDDING_GENERATION.value: WorkerConfig(
                    TaskType.EMBEDDING_GENERATION, 2, ResourceType.CPU, 1
                ),
                TaskType.THUMBNAIL_GENERATION.value: WorkerConfig(
                    TaskType.THUMBNAIL_GENERATION, 1, ResourceType.CPU, 4
                ),
            },
            task_settings=TaskSettings(
                max_concurrent_videos=10,
                frame_sampling_interval=60,
                face_sampling_interval_seconds=10.0,
            ),
        )

        # Visual First Profile
        visual_first = ProcessingProfile(
            name="visual_first",
            description="Prioritize object and face detection for visual navigation",
            worker_configs={
                TaskType.HASH.value: WorkerConfig(
                    TaskType.HASH, 3, ResourceType.CPU, 1
                ),
                TaskType.TRANSCRIPTION.value: WorkerConfig(
                    TaskType.TRANSCRIPTION, 1, ResourceType.CPU, 3
                ),
                TaskType.SCENE_DETECTION.value: WorkerConfig(
                    TaskType.SCENE_DETECTION, 2, ResourceType.CPU, 2
                ),
                TaskType.OBJECT_DETECTION.value: WorkerConfig(
                    TaskType.OBJECT_DETECTION, 3, ResourceType.GPU, 1
                ),
                TaskType.FACE_DETECTION.value: WorkerConfig(
                    TaskType.FACE_DETECTION, 3, ResourceType.GPU, 1
                ),
                TaskType.OCR.value: WorkerConfig(TaskType.OCR, 2, ResourceType.GPU, 2),
                TaskType.PLACE_DETECTION.value: WorkerConfig(
                    TaskType.PLACE_DETECTION, 2, ResourceType.GPU, 2
                ),
                TaskType.TOPIC_EXTRACTION.value: WorkerConfig(
                    TaskType.TOPIC_EXTRACTION, 1, ResourceType.CPU, 4
                ),
                TaskType.EMBEDDING_GENERATION.value: WorkerConfig(
                    TaskType.EMBEDDING_GENERATION, 1, ResourceType.CPU, 3
                ),
                TaskType.THUMBNAIL_GENERATION.value: WorkerConfig(
                    TaskType.THUMBNAIL_GENERATION, 2, ResourceType.CPU, 2
                ),
            },
            task_settings=TaskSettings(
                max_concurrent_videos=3,
                frame_sampling_interval=15,
                face_sampling_interval_seconds=2.0,
            ),
        )

        # Low Resource Profile
        low_resource = ProcessingProfile(
            name="low_resource",
            description="Minimal resource usage for background processing",
            worker_configs={
                TaskType.HASH.value: WorkerConfig(
                    TaskType.HASH, 2, ResourceType.CPU, 1
                ),
                TaskType.TRANSCRIPTION.value: WorkerConfig(
                    TaskType.TRANSCRIPTION, 1, ResourceType.CPU, 2
                ),
                TaskType.SCENE_DETECTION.value: WorkerConfig(
                    TaskType.SCENE_DETECTION, 1, ResourceType.CPU, 3
                ),
                TaskType.OBJECT_DETECTION.value: WorkerConfig(
                    TaskType.OBJECT_DETECTION, 1, ResourceType.GPU, 3
                ),
                TaskType.FACE_DETECTION.value: WorkerConfig(
                    TaskType.FACE_DETECTION, 1, ResourceType.GPU, 3
                ),
                TaskType.OCR.value: WorkerConfig(TaskType.OCR, 1, ResourceType.GPU, 4),
                TaskType.PLACE_DETECTION.value: WorkerConfig(
                    TaskType.PLACE_DETECTION, 1, ResourceType.GPU, 4
                ),
                TaskType.TOPIC_EXTRACTION.value: WorkerConfig(
                    TaskType.TOPIC_EXTRACTION, 1, ResourceType.CPU, 4
                ),
                TaskType.EMBEDDING_GENERATION.value: WorkerConfig(
                    TaskType.EMBEDDING_GENERATION, 1, ResourceType.CPU, 2
                ),
                TaskType.THUMBNAIL_GENERATION.value: WorkerConfig(
                    TaskType.THUMBNAIL_GENERATION, 1, ResourceType.CPU, 4
                ),
            },
            task_settings=TaskSettings(
                max_concurrent_videos=1,
                frame_sampling_interval=120,
                face_sampling_interval_seconds=30.0,
            ),
        )

        # Add all default profiles
        self.profiles[balanced.name] = balanced
        self.profiles[search_first.name] = search_first
        self.profiles[visual_first.name] = visual_first
        self.profiles[low_resource.name] = low_resource

        logger.info("Loaded 4 default processing profiles")


def create_profile_from_config(config_data: dict[str, Any]) -> ProcessingProfile:
    """Create a processing profile from configuration data."""
    return ProcessingProfile.from_dict(config_data)
