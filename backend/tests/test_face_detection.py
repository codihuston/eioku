"""Tests for face detection service and task handler."""

import uuid
from unittest.mock import Mock, patch

import pytest

from src.domain.models import Face, Task, Video
from src.services.face_detection_service import (
    FaceDetectionError,
    FaceDetectionService,
)
from src.services.face_detection_task_handler import FaceDetectionTaskHandler
from src.services.task_orchestration import TaskType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_yolo_face_model():
    """Create a mock YOLO face model."""
    model = Mock()
    model.names = {0: "face"}  # Face model typically has single class
    return model


@pytest.fixture
def mock_av_container():
    """Create a mock PyAV container."""
    container = Mock()
    stream = Mock()
    stream.average_rate = 30.0
    stream.frames = 90
    stream.codec_context.name = "h264"
    container.streams.video = [stream]
    return container


@pytest.fixture
def mock_frame():
    """Create a mock PyAV frame."""
    frame = Mock()
    frame.to_ndarray.return_value = Mock()  # numpy array
    return frame


@pytest.fixture
def mock_face_detection_box():
    """Create a mock YOLO face detection box."""
    box = Mock()
    box.cls = [0]  # face class
    box.xyxy = [Mock(tolist=Mock(return_value=[50.0, 60.0, 150.0, 200.0]))]
    box.conf = [0.92]
    return box


# ============================================================================
# FaceDetectionService Initialization Tests
# ============================================================================


class TestFaceDetectionServiceInit:
    """Tests for FaceDetectionService initialization."""

    @patch("ultralytics.YOLO")
    def test_initialization_success(self, mock_yolo_class):
        """Test successful service initialization."""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model

        service = FaceDetectionService(model_name="yolov8n-face.pt")

        assert service.model_name == "yolov8n-face.pt"
        assert service.model == mock_model
        mock_yolo_class.assert_called_once_with("yolov8n-face.pt")

    @patch("ultralytics.YOLO")
    def test_initialization_with_different_models(self, mock_yolo_class):
        """Test initialization with different face model variants."""
        model_variants = [
            "yolov8n-face.pt",
            "yolov8s-face.pt",
            "yolov8m-face.pt",
        ]

        for model_name in model_variants:
            mock_yolo_class.reset_mock()
            service = FaceDetectionService(model_name=model_name)
            assert service.model_name == model_name
            mock_yolo_class.assert_called_with(model_name)

    @patch("ultralytics.YOLO")
    def test_initialization_model_load_failure(self, mock_yolo_class):
        """Test initialization when model fails to load."""
        mock_yolo_class.side_effect = Exception("Model file not found")

        with pytest.raises(FaceDetectionError) as exc_info:
            FaceDetectionService(model_name="nonexistent.pt")

        assert "Failed to load YOLO face model" in str(exc_info.value)


# ============================================================================
# FaceDetectionService Detection Tests
# ============================================================================


class TestFaceDetectionService:
    """Tests for FaceDetectionService face detection."""

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_success(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test successful face detection."""
        # Setup mocks
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        # Create mock detection
        mock_box = Mock()
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[50.0, 60.0, 150.0, 200.0]))]
        mock_box.conf = [0.92]

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        # Setup frame iteration (3 frames, sample every 30)
        mock_av_container.decode.return_value = [mock_frame] * 90

        service = FaceDetectionService()
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=30,
        )

        # Should return single Face object with all detections
        assert len(faces) == 1
        face = faces[0]
        assert face.video_id == "test-video-id"
        assert face.person_id is None  # No clustering yet
        assert len(face.timestamps) == 3  # 3 sampled frames
        assert len(face.bounding_boxes) == 3
        assert face.confidence > 0  # Average confidence

    @patch("ultralytics.YOLO")
    def test_detect_faces_video_not_found(self, mock_yolo_class):
        """Test detection with non-existent video file."""
        mock_yolo_class.return_value = Mock()
        service = FaceDetectionService()

        with pytest.raises(FaceDetectionError) as exc_info:
            service.detect_faces_in_video(
                video_path="/nonexistent/video.mp4",
                video_id="test-id",
                sample_rate=30,
            )

        assert "Video file not found" in str(exc_info.value)

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_no_faces_found(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test detection when no faces are found."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        # No detections
        mock_result = Mock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        mock_av_container.decode.return_value = [mock_frame] * 90

        service = FaceDetectionService()
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=30,
        )

        # Should return empty list when no faces detected
        assert len(faces) == 0

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_multiple_faces_per_frame(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test detection with multiple faces in single frame."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        # Create 3 face detections in one frame
        mock_boxes = []
        for i in range(3):
            box = Mock()
            box.xyxy = [
                Mock(
                    tolist=Mock(
                        return_value=[
                            50.0 + i * 100,
                            60.0,
                            150.0 + i * 100,
                            200.0,
                        ]
                    )
                )
            ]
            box.conf = [0.90 + i * 0.02]
            mock_boxes.append(box)

        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        mock_av_container.decode.return_value = [mock_frame] * 30

        service = FaceDetectionService()
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=30,
        )

        # All faces grouped into single Face object
        assert len(faces) == 1
        face = faces[0]
        assert len(face.timestamps) == 3  # 3 detections from 1 sampled frame
        assert len(face.bounding_boxes) == 3

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_confidence_calculation(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test average confidence calculation."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        # Create detections with known confidences
        confidences = [0.90, 0.85, 0.95]

        # Create separate mock results for each call
        mock_results = []
        for conf in confidences:
            box = Mock()
            box.xyxy = [Mock(tolist=Mock(return_value=[50.0, 60.0, 150.0, 200.0]))]
            box.conf = [conf]

            result = Mock()
            result.boxes = [box]
            mock_results.append([result])

        # Set side_effect to return different results for each call
        mock_model.side_effect = mock_results

        # Create frames
        frames = []
        for _ in confidences:
            frame = Mock()
            frame.to_ndarray.return_value = Mock()
            frames.append(frame)

        mock_av_container.decode.return_value = frames

        service = FaceDetectionService()
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=1,  # Sample every frame
        )

        # Check average confidence
        assert len(faces) == 1
        expected_avg = sum(confidences) / len(confidences)
        assert abs(faces[0].confidence - expected_avg) < 0.01

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_person_id_is_null(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test that person_id is None (no clustering in Phase 1)."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        mock_box = Mock()
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[50.0, 60.0, 150.0, 200.0]))]
        mock_box.conf = [0.92]

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        mock_av_container.decode.return_value = [mock_frame] * 30

        service = FaceDetectionService()
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=30,
        )

        # person_id should be None (Phase 1 - no clustering)
        assert len(faces) == 1
        assert faces[0].person_id is None

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_timestamps_sorted(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test that timestamps are in chronological order."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        mock_box = Mock()
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[50.0, 60.0, 150.0, 200.0]))]
        mock_box.conf = [0.92]

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        mock_av_container.decode.return_value = [mock_frame] * 90

        service = FaceDetectionService()
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=30,
        )

        # Timestamps should be in order
        assert len(faces) == 1
        timestamps = faces[0].timestamps
        assert timestamps == sorted(timestamps)

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_bounding_boxes_format(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test bounding box format."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        mock_box = Mock()
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[50.0, 60.0, 150.0, 200.0]))]
        mock_box.conf = [0.92]

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        mock_av_container.decode.return_value = [mock_frame] * 30

        service = FaceDetectionService()
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=30,
        )

        # Check bounding box format
        assert len(faces) == 1
        bbox = faces[0].bounding_boxes[0]
        assert "frame" in bbox
        assert "timestamp" in bbox
        assert "bbox" in bbox
        assert "confidence" in bbox
        assert len(bbox["bbox"]) == 4  # [x1, y1, x2, y2]

    @patch("ultralytics.YOLO")
    @patch("av.open")
    @patch("pathlib.Path.exists")
    def test_detect_faces_frame_sampling(
        self,
        mock_path_exists,
        mock_av_open,
        mock_yolo_class,
        mock_av_container,
        mock_frame,
    ):
        """Test frame sampling with different sample rates."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_av_open.return_value = mock_av_container

        mock_box = Mock()
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[50.0, 60.0, 150.0, 200.0]))]
        mock_box.conf = [0.92]

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        # 90 frames at 30fps = 3 seconds
        mock_av_container.decode.return_value = [mock_frame] * 90

        service = FaceDetectionService()

        # Sample every 30 frames (1 per second at 30fps)
        faces = service.detect_faces_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=30,
        )

        # Should process 3 frames (0, 30, 60)
        assert len(faces) == 1
        assert len(faces[0].timestamps) == 3


# ============================================================================
# FaceDetectionTaskHandler Tests
# ============================================================================


class TestFaceDetectionTaskHandler:
    """Tests for FaceDetectionTaskHandler."""

    def test_handler_initialization(self):
        """Test handler initialization."""
        mock_repo = Mock()
        mock_service = Mock()

        handler = FaceDetectionTaskHandler(
            face_repository=mock_repo,
            detection_service=mock_service,
            model_name="yolov8n-face.pt",
            sample_rate=30,
        )

        assert handler.face_repository == mock_repo
        assert handler.detection_service == mock_service
        assert handler.model_name == "yolov8n-face.pt"
        assert handler.sample_rate == 30

    def test_process_face_detection_task_success(self):
        """Test successful face detection task processing."""
        mock_repo = Mock()
        mock_service = Mock()

        # Create mock face
        mock_face = Face(
            face_id=str(uuid.uuid4()),
            video_id="test-video-id",
            person_id=None,
            timestamps=[1.0, 2.0, 3.0],
            bounding_boxes=[
                {
                    "frame": 30,
                    "timestamp": 1.0,
                    "bbox": [50, 60, 150, 200],
                    "confidence": 0.92,
                }
            ],
            confidence=0.92,
        )

        mock_service.detect_faces_in_video.return_value = [mock_face]

        handler = FaceDetectionTaskHandler(
            face_repository=mock_repo,
            detection_service=mock_service,
            sample_rate=30,
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.FACE_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        result = handler.process_face_detection_task(task, video)

        assert result is True
        mock_service.detect_faces_in_video.assert_called_once()
        mock_repo.save.assert_called_once_with(mock_face)

    def test_process_face_detection_task_failure(self):
        """Test face detection task processing failure."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_faces_in_video.side_effect = Exception("Detection failed")

        handler = FaceDetectionTaskHandler(
            face_repository=mock_repo,
            detection_service=mock_service,
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.FACE_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        result = handler.process_face_detection_task(task, video)

        assert result is False
        mock_repo.save.assert_not_called()

    def test_get_detected_faces(self):
        """Test getting detected faces for a video."""
        mock_repo = Mock()
        mock_faces = [
            Face(
                face_id=str(uuid.uuid4()),
                video_id="test-video-id",
                person_id=None,
                timestamps=[1.0, 2.0],
                bounding_boxes=[],
                confidence=0.90,
            )
        ]
        mock_repo.find_by_video_id.return_value = mock_faces

        handler = FaceDetectionTaskHandler(
            face_repository=mock_repo,
            detection_service=Mock(),
        )

        faces = handler.get_detected_faces("test-video-id")

        assert faces == mock_faces
        mock_repo.find_by_video_id.assert_called_once_with("test-video-id")

    def test_get_detected_faces_empty(self):
        """Test getting faces when none exist."""
        mock_repo = Mock()
        mock_repo.find_by_video_id.return_value = []

        handler = FaceDetectionTaskHandler(
            face_repository=mock_repo,
            detection_service=Mock(),
        )

        faces = handler.get_detected_faces("test-video-id")

        assert faces == []

    def test_get_faces_by_person(self):
        """Test getting faces filtered by person ID."""
        mock_repo = Mock()
        mock_faces = [
            Face(
                face_id=str(uuid.uuid4()),
                video_id="test-video-id",
                person_id="person-1",
                timestamps=[1.0, 2.0],
                bounding_boxes=[],
                confidence=0.90,
            )
        ]
        mock_repo.find_by_person_id.return_value = mock_faces

        handler = FaceDetectionTaskHandler(
            face_repository=mock_repo,
            detection_service=Mock(),
        )

        faces = handler.get_faces_by_person("test-video-id", "person-1")

        assert faces == mock_faces
        mock_repo.find_by_person_id.assert_called_once_with("test-video-id", "person-1")

    def test_face_detection_task_with_custom_sample_rate(self):
        """Test task processing with custom sample rate."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_faces_in_video.return_value = []

        handler = FaceDetectionTaskHandler(
            face_repository=mock_repo,
            detection_service=mock_service,
            sample_rate=60,  # Custom sample rate
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.FACE_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        handler.process_face_detection_task(task, video)

        # Verify sample_rate was passed
        call_args = mock_service.detect_faces_in_video.call_args
        assert call_args[1]["sample_rate"] == 60
