"""Places detection task handler for video processing orchestration."""

from ..domain.models import Task, Video
from ..repositories.interfaces import PlaceRepository
from ..utils.print_logger import get_logger
from .places_detection_service import PlacesDetectionService

logger = get_logger(__name__)


class PlacesDetectionTaskHandler:
    """Handles places detection tasks in the orchestration system."""

    def __init__(
        self,
        place_repository: PlaceRepository,
        detection_service: PlacesDetectionService | None = None,
        model_path: str = "resnet18_places365.pth.tar",
        sample_rate: int = 30,
        top_k: int = 5,
    ):
        self.place_repository = place_repository
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.detection_service = detection_service or PlacesDetectionService(
            model_path=model_path
        )

    def process_places_detection_task(self, task: Task, video: Video) -> bool:
        """Process a places detection task for a video.

        Args:
            task: The places detection task to process
            video: The video to analyze

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting places detection for video {video.video_id}")

            # Detect places in video using configured sample rate and top-K
            places = self.detection_service.detect_places_in_video(
                video_path=video.file_path,
                video_id=video.video_id,
                sample_rate=self.sample_rate,
                top_k=self.top_k,
            )

            logger.info(f"Detected {len(places)} unique place types")

            # Save detected places to database
            saved_count = 0
            for place in places:
                self.place_repository.save(place)
                saved_count += 1
                logger.debug(
                    f"Saved place '{place.label}' with "
                    f"{place.get_occurrence_count()} occurrences "
                    f"(confidence: {place.confidence:.3f})"
                )

            logger.info(
                f"Places detection complete for video {video.video_id}. "
                f"Saved {saved_count} place types"
            )
            return True

        except Exception as e:
            logger.error(f"Places detection failed for video {video.video_id}: {e}")
            return False

    def get_detected_places(self, video_id: str) -> list:
        """Get all detected places for a video.

        Args:
            video_id: Video ID

        Returns:
            List of detected places
        """
        return self.place_repository.find_by_video_id(video_id)

    def get_places_by_label(self, video_id: str, label: str) -> list:
        """Get detected places filtered by label.

        Args:
            video_id: Video ID
            label: Place label to filter by

        Returns:
            List of detected places with the specified label
        """
        return self.place_repository.find_by_label(video_id, label)
