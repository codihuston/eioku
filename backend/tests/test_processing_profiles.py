"""Tests for processing profile configuration."""

import tempfile
from pathlib import Path

from src.services.processing_profiles import (
    ProcessingProfile,
    ProfileManager,
    TaskSettings,
    create_profile_from_config,
)
from src.services.task_orchestration import TaskType
from src.services.worker_pool_manager import ResourceType, WorkerConfig


class TestTaskSettings:
    """Test task settings functionality."""

    def test_task_settings_defaults(self):
        """Test default task settings."""
        settings = TaskSettings()

        assert settings.max_concurrent_videos == 5
        assert settings.frame_sampling_interval == 30
        assert settings.face_sampling_interval_seconds == 5.0
        assert settings.transcription_model == "large-v3"
        assert settings.object_detection_model == "yolov8n.pt"
        assert settings.face_detection_model == "yolov8n-face.pt"

    def test_task_settings_custom(self):
        """Test custom task settings."""
        settings = TaskSettings(
            max_concurrent_videos=10,
            frame_sampling_interval=60,
            face_sampling_interval_seconds=2.0,
        )

        assert settings.max_concurrent_videos == 10
        assert settings.frame_sampling_interval == 60
        assert settings.face_sampling_interval_seconds == 2.0


class TestProcessingProfile:
    """Test processing profile functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.worker_configs = {
            TaskType.HASH.value: WorkerConfig(TaskType.HASH, 4, ResourceType.CPU, 1),
            TaskType.TRANSCRIPTION.value: WorkerConfig(
                TaskType.TRANSCRIPTION, 2, ResourceType.CPU, 2
            ),
        }

        self.task_settings = TaskSettings(
            max_concurrent_videos=5, frame_sampling_interval=30
        )

        self.profile = ProcessingProfile(
            name="test_profile",
            description="Test profile",
            worker_configs=self.worker_configs,
            task_settings=self.task_settings,
        )

    def test_profile_creation(self):
        """Test profile creation."""
        assert self.profile.name == "test_profile"
        assert self.profile.description == "Test profile"
        assert len(self.profile.worker_configs) == 2
        assert self.profile.task_settings.max_concurrent_videos == 5

    def test_profile_to_dict(self):
        """Test profile serialization to dictionary."""
        data = self.profile.to_dict()

        assert data["name"] == "test_profile"
        assert data["description"] == "Test profile"
        assert "workers" in data
        assert "task_settings" in data

        # Check worker data
        workers = data["workers"]
        assert TaskType.HASH.value in workers
        hash_worker = workers[TaskType.HASH.value]
        assert hash_worker["count"] == 4
        assert hash_worker["resource"] == "cpu"
        assert hash_worker["priority"] == 1

    def test_profile_from_dict(self):
        """Test profile deserialization from dictionary."""
        data = {
            "name": "test_profile",
            "description": "Test profile",
            "workers": {
                TaskType.HASH.value: {"count": 4, "resource": "cpu", "priority": 1},
                TaskType.TRANSCRIPTION.value: {
                    "count": 2,
                    "resource": "cpu",
                    "priority": 2,
                },
            },
            "task_settings": {
                "max_concurrent_videos": 5,
                "frame_sampling_interval": 30,
                "face_sampling_interval_seconds": 5.0,
                "transcription_model": "large-v3",
                "object_detection_model": "yolov8n.pt",
                "face_detection_model": "yolov8n-face.pt",
            },
        }

        profile = ProcessingProfile.from_dict(data)

        assert profile.name == "test_profile"
        assert profile.description == "Test profile"
        assert len(profile.worker_configs) == 2

        # Check worker config
        hash_config = profile.worker_configs[TaskType.HASH.value]
        assert hash_config.worker_count == 4
        assert hash_config.resource_type == ResourceType.CPU
        assert hash_config.priority == 1

    def test_profile_roundtrip_serialization(self):
        """Test profile serialization roundtrip."""
        # Convert to dict and back
        data = self.profile.to_dict()
        restored_profile = ProcessingProfile.from_dict(data)

        assert restored_profile.name == self.profile.name
        assert restored_profile.description == self.profile.description
        assert len(restored_profile.worker_configs) == len(self.profile.worker_configs)


class TestProfileManager:
    """Test profile manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ProfileManager(config_dir=self.temp_dir)

    def test_manager_initialization(self):
        """Test profile manager initialization."""
        assert len(self.manager.profiles) == 4  # 4 default profiles
        assert "balanced" in self.manager.profiles
        assert "search_first" in self.manager.profiles
        assert "visual_first" in self.manager.profiles
        assert "low_resource" in self.manager.profiles

    def test_get_profile(self):
        """Test getting a profile."""
        profile = self.manager.get_profile("balanced")

        assert profile.name == "balanced"
        assert "Even resource distribution" in profile.description
        assert TaskType.HASH.value in profile.worker_configs

    def test_get_nonexistent_profile_raises_error(self):
        """Test getting nonexistent profile raises error."""
        try:
            self.manager.get_profile("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)

    def test_list_profiles(self):
        """Test listing profiles."""
        profiles = self.manager.list_profiles()

        assert len(profiles) == 4
        assert "balanced" in profiles
        assert "search_first" in profiles
        assert "visual_first" in profiles
        assert "low_resource" in profiles

        # Check descriptions
        assert "Even resource distribution" in profiles["balanced"]
        assert "searchable quickly" in profiles["search_first"]
        assert "visual navigation" in profiles["visual_first"]
        assert "Minimal resource" in profiles["low_resource"]

    def test_add_custom_profile(self):
        """Test adding a custom profile."""
        custom_profile = ProcessingProfile(
            name="custom",
            description="Custom profile",
            worker_configs={
                TaskType.HASH.value: WorkerConfig(TaskType.HASH, 1, ResourceType.CPU, 1)
            },
            task_settings=TaskSettings(),
        )

        self.manager.add_profile(custom_profile)

        assert "custom" in self.manager.profiles
        retrieved = self.manager.get_profile("custom")
        assert retrieved.name == "custom"

    def test_save_and_load_profile(self):
        """Test saving and loading profiles."""
        # Get a profile to save
        profile = self.manager.get_profile("balanced")

        # Save to file
        file_path = Path(self.temp_dir) / "test_profile.json"
        self.manager.save_profile("balanced", str(file_path))

        assert file_path.exists()

        # Create new manager and load profile
        new_manager = ProfileManager(config_dir=self.temp_dir)
        loaded_profile = new_manager.load_profile(str(file_path))

        assert loaded_profile.name == profile.name
        assert loaded_profile.description == profile.description
        assert len(loaded_profile.worker_configs) == len(profile.worker_configs)

    def test_save_nonexistent_profile_raises_error(self):
        """Test saving nonexistent profile raises error."""
        try:
            self.manager.save_profile("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)

    def test_load_nonexistent_file_raises_error(self):
        """Test loading nonexistent file raises error."""
        try:
            self.manager.load_profile("nonexistent.json")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass


class TestDefaultProfiles:
    """Test default profile configurations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ProfileManager()

    def test_balanced_profile(self):
        """Test balanced profile configuration."""
        profile = self.manager.get_profile("balanced")

        # Check hash workers (should be highest count for prerequisite)
        hash_config = profile.worker_configs[TaskType.HASH.value]
        assert hash_config.worker_count == 4
        assert hash_config.priority == 1  # Critical

        # Check transcription workers
        trans_config = profile.worker_configs[TaskType.TRANSCRIPTION.value]
        assert trans_config.worker_count == 2
        assert trans_config.priority == 2  # High

        # Check task settings
        assert profile.task_settings.max_concurrent_videos == 5

    def test_search_first_profile(self):
        """Test search first profile configuration."""
        profile = self.manager.get_profile("search_first")

        # Should prioritize transcription and embedding
        trans_config = profile.worker_configs[TaskType.TRANSCRIPTION.value]
        embed_config = profile.worker_configs[TaskType.EMBEDDING_GENERATION.value]

        assert trans_config.worker_count == 4  # More than balanced
        assert trans_config.priority == 1  # Critical
        assert embed_config.priority == 1  # Critical

        # Visual tasks should have lower priority
        obj_config = profile.worker_configs[TaskType.OBJECT_DETECTION.value]
        assert obj_config.worker_count == 1  # Less than balanced
        assert obj_config.priority == 4  # Low

    def test_visual_first_profile(self):
        """Test visual first profile configuration."""
        profile = self.manager.get_profile("visual_first")

        # Should prioritize visual tasks
        obj_config = profile.worker_configs[TaskType.OBJECT_DETECTION.value]
        face_config = profile.worker_configs[TaskType.FACE_DETECTION.value]

        assert obj_config.worker_count == 3  # More than balanced
        assert obj_config.priority == 1  # Critical
        assert face_config.worker_count == 3  # More than balanced
        assert face_config.priority == 1  # Critical

        # Should have more frequent sampling
        assert profile.task_settings.frame_sampling_interval == 15  # More frequent
        assert (
            profile.task_settings.face_sampling_interval_seconds == 2.0
        )  # More frequent

    def test_low_resource_profile(self):
        """Test low resource profile configuration."""
        profile = self.manager.get_profile("low_resource")

        # Should have minimal workers
        for task_type, config in profile.worker_configs.items():
            if task_type == TaskType.HASH.value:
                assert config.worker_count == 2  # Still need some hash workers
            else:
                assert config.worker_count == 1  # Minimal for others

        # Should process fewer videos concurrently
        assert profile.task_settings.max_concurrent_videos == 1

        # Should have less frequent sampling
        assert profile.task_settings.frame_sampling_interval == 120  # Less frequent
        assert (
            profile.task_settings.face_sampling_interval_seconds == 30.0
        )  # Less frequent


def test_create_profile_from_config():
    """Test creating profile from config data."""
    config_data = {
        "name": "test",
        "description": "Test profile",
        "workers": {
            TaskType.HASH.value: {"count": 2, "resource": "cpu", "priority": 1}
        },
        "task_settings": {
            "max_concurrent_videos": 3,
            "frame_sampling_interval": 45,
            "face_sampling_interval_seconds": 7.5,
            "transcription_model": "large-v3",
            "object_detection_model": "yolov8n.pt",
            "face_detection_model": "yolov8n-face.pt",
        },
    }

    profile = create_profile_from_config(config_data)

    assert profile.name == "test"
    assert profile.description == "Test profile"
    assert len(profile.worker_configs) == 1
    assert profile.task_settings.max_concurrent_videos == 3
