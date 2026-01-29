"""Video file discovery service."""

from pathlib import Path
from uuid import uuid4

from ..domain.models import PathConfig, Task, Video
from ..domain.task_registry import (
    is_language_optional,
    is_language_required,
)
from ..repositories.interfaces import VideoRepository
from ..repositories.task_repository import SQLAlchemyTaskRepository
from ..utils.print_logger import get_logger
from .job_producer import JobProducer
from .path_config_manager import PathConfigManager

# Configure logging
logger = get_logger(__name__)

# Active task types (subset of TASK_REGISTRY that are currently enabled)
ACTIVE_TASK_TYPES = [
    "object_detection",
    "face_detection",
    "transcription",
    "ocr",
    "place_detection",
    "scene_detection",
    "metadata_extraction",
]


class VideoDiscoveryService:
    """Service for discovering video files in configured paths."""

    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}

    def __init__(
        self,
        path_config_manager: PathConfigManager,
        video_repository: VideoRepository,
        job_producer: JobProducer | None = None,
    ):
        self.path_config_manager = path_config_manager
        self.video_repository = video_repository
        self.job_producer = job_producer

    def discover_videos(self) -> list[Video]:
        """Discover all video files in configured paths."""
        logger.info("Starting discovery...")
        discovered_videos = []
        path_configs = self.path_config_manager.list_paths()
        logger.info(f"Found {len(path_configs)} configured paths")

        for path_config in path_configs:
            logger.info(f"Scanning path: {path_config.path}")
            videos = self._scan_path(path_config)
            logger.info(f"Found {len(videos)} videos in {path_config.path}")
            discovered_videos.extend(videos)

        logger.info(f"Total discovered videos: {len(discovered_videos)}")
        return discovered_videos

    def _scan_path(self, path_config: PathConfig) -> list[Video]:
        """Scan a single path configuration for video files."""
        videos = []
        path = Path(path_config.path)

        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return videos

        # Use glob patterns for each supported format (much more efficient)
        # Check both lowercase and uppercase extensions for case-insensitive matching
        for extension in self.SUPPORTED_FORMATS:
            for ext_variant in [extension, extension.upper()]:
                pattern = (
                    f"**/*{ext_variant}" if path_config.recursive else f"*{ext_variant}"
                )
                logger.debug(f"Scanning with pattern: {pattern}")

                glob_method = path.rglob if path_config.recursive else path.glob
                for video_file in glob_method(f"*{ext_variant}"):
                    if video_file.is_file():
                        video = self._create_video_from_file(video_file)
                        if video:
                            videos.append(video)

        return videos

    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS

    def _create_video_from_file(self, file_path: Path) -> Video | None:
        """Create Video domain object from file path."""
        try:
            logger.debug(f"Creating video from file: {file_path}")

            # Check if video already exists in database
            existing = self.video_repository.find_by_path(str(file_path))
            if existing:
                logger.debug(f"Video already exists: {existing.video_id}")
                return existing

            # Get file stats
            stat = file_path.stat()
            logger.debug(
                f"File stats - size: {stat.st_size}, modified: {stat.st_mtime}"
            )

            # Compute file hash during discovery
            from .file_hash_service import FileHashService

            hash_service = FileHashService()
            try:
                file_hash = hash_service.calculate_hash(str(file_path))
                logger.info(f"Computed file hash for {file_path.name}: {file_hash}")
            except Exception as e:
                logger.error(f"Failed to compute file hash for {file_path}: {e}")
                file_hash = None

            # Create new video
            import uuid
            from datetime import datetime

            video = Video(
                video_id=str(uuid.uuid4()),
                file_path=str(file_path),
                filename=file_path.name,
                file_hash=file_hash,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                status="discovered",
                file_size=stat.st_size,
            )
            logger.debug(f"Created video object: {video.video_id}")

            # Save to database
            logger.debug("Attempting to save video to database...")
            saved_video = self.video_repository.save(video)
            logger.info(f"Video saved successfully: {saved_video.video_id}")
            return saved_video

        except Exception as e:
            # Log the error for debugging
            logger.error(f"Error creating video from {file_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def validate_existing_videos(self) -> list[Video]:
        """Validate existing videos exist on filesystem and clean up orphaned tasks."""
        missing_videos = []
        # Check all videos that have been discovered or completed
        all_videos = []
        all_videos.extend(self.video_repository.find_by_status("discovered"))
        all_videos.extend(self.video_repository.find_by_status("completed"))
        all_videos.extend(self.video_repository.find_by_status("hashed"))
        all_videos.extend(self.video_repository.find_by_status("processing"))

        for video in all_videos:
            if not Path(video.file_path).exists():
                logger.info(
                    f"Video missing from filesystem: {video.filename} "
                    f"at {video.file_path}"
                )

                # Delete the video from database
                self.video_repository.delete(video.video_id)
                missing_videos.append(video)

                logger.info(f"Removed missing video {video.video_id} from database")

        return missing_videos

    async def discover_and_queue_tasks(self, video_path: str) -> str:
        """Discover video and auto-create tasks for all ML operations.

        This method:
        1. Checks if video already exists in database
        2. Creates video record if new
        3. Auto-creates tasks based on language configuration:
           - OCR: One task per configured language (required)
           - Transcription: NULL (auto-detect) or one per language (optional)
           - Others: Single task with NULL language
        4. Enqueues tasks to appropriate queues via JobProducer

        Args:
            video_path: Path to video file

        Returns:
            video_id of the discovered/existing video

        Raises:
            ValueError: If video file doesn't exist or JobProducer not initialized
        """
        if not self.job_producer:
            raise ValueError("JobProducer not initialized. Cannot auto-queue tasks.")

        # Check if video file exists
        video_file = Path(video_path)
        if not video_file.exists():
            raise ValueError(f"Video file not found: {video_path}")

        logger.info(f"Discovering and queueing tasks for: {video_path}")

        # Check if video already exists
        existing = self.video_repository.find_by_path(video_path)
        if existing:
            logger.info(f"Video already exists: {existing.video_id}")
            video = existing
        else:
            # Create video record
            video = self._create_video_from_file(video_file)
            if not video:
                raise ValueError(f"Failed to create video record for: {video_path}")
            logger.info(f"Created video record: {video.video_id}")

        task_repo = SQLAlchemyTaskRepository(self.video_repository.session)

        for task_type in ACTIVE_TASK_TYPES:
            # Get default config for task type
            config = self._get_default_config(task_type)

            if is_language_required(task_type):
                # OCR: Create one task per configured language
                languages = config.get("languages", ["en"])
                for lang in languages:
                    await self._create_task_if_not_exists(
                        task_repo=task_repo,
                        video=video,
                        video_path=video_path,
                        task_type=task_type,
                        language=lang,
                        config=config,
                    )
            elif is_language_optional(task_type):
                # Transcription: Check if languages are configured
                languages = config.get("languages")
                if languages and isinstance(languages, list) and len(languages) > 0:
                    # Create one task per configured language
                    for lang in languages:
                        await self._create_task_if_not_exists(
                            task_repo=task_repo,
                            video=video,
                            video_path=video_path,
                            task_type=task_type,
                            language=lang,
                            config=config,
                        )
                else:
                    # Auto-detect mode: single task with NULL language
                    await self._create_task_if_not_exists(
                        task_repo=task_repo,
                        video=video,
                        video_path=video_path,
                        task_type=task_type,
                        language=None,
                        config=config,
                    )
            else:
                # Language-agnostic tasks (face_detection, object_detection, etc.)
                await self._create_task_if_not_exists(
                    task_repo=task_repo,
                    video=video,
                    video_path=video_path,
                    task_type=task_type,
                    language=None,
                    config=config,
                )

        logger.info(
            f"Successfully discovered and queued all tasks for video {video.video_id}"
        )
        return video.video_id

    async def _create_task_if_not_exists(
        self,
        task_repo: SQLAlchemyTaskRepository,
        video: Video,
        video_path: str,
        task_type: str,
        language: str | None,
        config: dict,
    ) -> bool:
        """Create a task if it doesn't already exist.

        Args:
            task_repo: Task repository instance
            video: Video domain object
            video_path: Path to video file
            task_type: Type of task to create
            language: Language for the task (None for language-agnostic)
            config: Configuration dictionary for the task

        Returns:
            True if task was created, False if it already existed
        """
        # Check if task already exists for this video, type, and language
        existing_task = task_repo.find_by_video_type_language(
            video.video_id, task_type, language
        )
        if existing_task:
            lang_str = f" ({language})" if language else ""
            logger.info(
                f"Task already exists for video {video.video_id} "
                f"({task_type}{lang_str}), skipping creation"
            )
            return False

        task_id = str(uuid4())

        # Build task-specific config with language
        task_config = config.copy()
        if language:
            # For OCR: use singular 'language' key
            task_config["language"] = language
            # Remove languages list to avoid confusion
            task_config.pop("languages", None)

        # Create task record in PostgreSQL
        task = Task(
            task_id=task_id,
            video_id=video.video_id,
            task_type=task_type,
            language=language,
            status="pending",
            priority=1,
        )
        try:
            task_repo.save(task)
            lang_str = f" ({language})" if language else ""
            logger.info(
                f"Created task record {task_id} ({task_type}{lang_str}) for "
                f"video {video.video_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to create task record {task_id} ({task_type}): {e}",
                exc_info=True,
            )
            raise

        # Enqueue job to Redis
        try:
            lang_str = f" ({language})" if language else ""
            logger.info(
                f"Enqueueing task {task_id} ({task_type}{lang_str}) "
                f"with config: {task_config}"
            )
            await self.job_producer.enqueue_task(
                task_id=task_id,
                task_type=task_type,
                video_id=video.video_id,
                video_path=video_path,
                config=task_config,
            )
            logger.info(
                f"Enqueued task {task_id} ({task_type}{lang_str}) "
                f"for video {video.video_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to enqueue task {task_id} ({task_type}): {e}",
                exc_info=True,
            )
            raise

        return True

    def _get_default_config(self, task_type: str) -> dict:
        """Get default configuration for task type.

        Args:
            task_type: Type of task (e.g., 'object_detection')

        Returns:
            Dictionary with default configuration for the task type
        """
        # Load from config/content_creator.json if available
        config_file = (
            Path(__file__).parent.parent.parent / "config" / "content_creator.json"
        )
        task_settings = {}

        if config_file.exists():
            try:
                import json

                with open(config_file) as f:
                    config_data = json.load(f)
                    all_settings = config_data.get("task_settings", {})
                    # Get task-specific settings if available
                    task_settings = all_settings.get(task_type, {})
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")

        # Default configurations with fallbacks
        defaults = {
            "object_detection": {
                "model_name": "yolov8n.pt",
                "frame_interval": 3,
                "confidence_threshold": 0.7,
                "model_profile": "balanced",
            },
            "face_detection": {
                "model_name": "yolov8n-face.pt",
                "frame_interval": 3,
                "confidence_threshold": 0.7,
            },
            "transcription": {
                "model_name": "large-v3",
                "language": None,
                "vad_filter": True,
            },
            "ocr": {
                "frame_interval": 2,
                "languages": ["en"],
                "use_gpu": True,
            },
            "place_detection": {
                "frame_interval": 2,
                "top_k": 5,
            },
            "scene_detection": {
                "threshold": 0.7,
                "min_scene_length": 0.6,
            },
            "metadata_extraction": {},
        }

        # Get default config for task type
        config = defaults.get(task_type, {}).copy()

        # Override with loaded settings
        if task_settings:
            # Map sampling_interval_seconds to frame_interval for consistency
            if "sampling_interval_seconds" in task_settings:
                config["frame_interval"] = task_settings["sampling_interval_seconds"]
            # Merge other settings
            config.update(task_settings)

        return config
