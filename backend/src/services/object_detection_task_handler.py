"""Object detection task handler for video processing orchestration."""

from ..domain.models import Task, Video
from ..repositories.interfaces import ObjectRepository
from ..utils.print_logger import get_logger
from .object_detection_service import ObjectDetectionService

logger = get_logger(__name__)


class ObjectDetectionTaskHandler:
    """Handles object detection tasks in the orchestration system."""

    def __init__(
        self,
        object_repository: ObjectRepository,
        detection_service: ObjectDetectionService | None = None,
        model_name: str = "yolov8n.pt",
        sample_rate: int = 30,
    ):
        self.object_repository = object_repository
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.detection_service = detection_service or ObjectDetectionService(
            model_name=model_name
        )

    def process_object_detection_task(self, task: Task, video: Video) -> bool:
        """Process an object detection task for a video.

        Args:
            task: The object detection task to process
            video: The video to analyze

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting object detection for video {video.video_id}")

            # Detect objects in video using configured sample rate
            objects = self.detection_service.detect_objects_in_video(
                video_path=video.file_path,
                video_id=video.video_id,
                sample_rate=self.sample_rate,
            )

            logger.info(f"Detected {len(objects)} unique object types")

            # Save detected objects to database
            saved_count = 0
            for obj in objects:
                self.object_repository.save(obj)
                saved_count += 1
                logger.debug(
                    f"Saved object '{obj.label}' with "
                    f"{obj.get_occurrence_count()} occurrences"
                )

            logger.info(
                f"Object detection complete for video {video.video_id}. "
                f"Saved {saved_count} object types"
            )
            return True

        except Exception as e:
            logger.error(f"Object detection failed for video {video.video_id}: {e}")
            return False

    def get_detected_objects(self, video_id: str) -> list:
        """Get all detected objects for a video.

        Args:
            video_id: Video ID

        Returns:
            List of detected objects
        """
        return self.object_repository.find_by_video_id(video_id)

    def get_objects_by_label(self, video_id: str, label: str) -> list:
        """Get detected objects filtered by label.

        Args:
            video_id: Video ID
            label: Object label to filter by

        Returns:
            List of detected objects with the specified label
        """
        return self.object_repository.find_by_label(video_id, label)
