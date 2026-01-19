"""Face detection task handler for video processing orchestration."""

from ..domain.models import Task, Video
from ..repositories.interfaces import FaceRepository
from ..utils.print_logger import get_logger
from .face_detection_service import FaceDetectionService

logger = get_logger(__name__)


class FaceDetectionTaskHandler:
    """Handles face detection tasks in the orchestration system."""

    def __init__(
        self,
        face_repository: FaceRepository,
        detection_service: FaceDetectionService | None = None,
        model_name: str = "yolov8n-face.pt",
        sample_rate: int = 30,
    ):
        self.face_repository = face_repository
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.detection_service = detection_service or FaceDetectionService(
            model_name=model_name
        )

    def process_face_detection_task(self, task: Task, video: Video) -> bool:
        """Process a face detection task for a video.

        Args:
            task: The face detection task to process
            video: The video to analyze

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting face detection for video {video.video_id}")

            # Detect faces in video using configured sample rate
            faces = self.detection_service.detect_faces_in_video(
                video_path=video.file_path,
                video_id=video.video_id,
                sample_rate=self.sample_rate,
            )

            logger.info(f"Detected {len(faces)} face groups")

            # Save detected faces to database
            saved_count = 0
            for face in faces:
                self.face_repository.save(face)
                saved_count += 1
                logger.debug(
                    f"Saved face group with "
                    f"{face.get_occurrence_count()} occurrences, "
                    f"avg confidence: {face.confidence:.2f}"
                )

            logger.info(
                f"Face detection complete for video {video.video_id}. "
                f"Saved {saved_count} face groups"
            )
            return True

        except Exception as e:
            logger.error(f"Face detection failed for video {video.video_id}: {e}")
            return False

    def get_detected_faces(self, video_id: str) -> list:
        """Get all detected faces for a video.

        Args:
            video_id: Video ID

        Returns:
            List of detected faces
        """
        return self.face_repository.find_by_video_id(video_id)

    def get_faces_by_person(self, video_id: str, person_id: str) -> list:
        """Get detected faces filtered by person ID.

        Args:
            video_id: Video ID
            person_id: Person ID to filter by

        Returns:
            List of detected faces with the specified person ID
        """
        return self.face_repository.find_by_person_id(video_id, person_id)
