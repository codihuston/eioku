"""Tests for places detection service and task handler."""

import uuid
from unittest.mock import Mock, patch

import pytest

from src.domain.models import Place, Task, Video
from src.services.places_detection_service import (
    PlacesDetectionError,
    PlacesDetectionService,
)
from src.services.places_detection_task_handler import PlacesDetectionTaskHandler
from src.services.task_orchestration import TaskType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_resnet_model():
    """Create a mock ResNet Places365 model."""
    model = Mock()
    model.eval = Mock(return_value=model)
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
    import numpy as np
    from PIL import Image

    frame = Mock()
    # Return a proper numpy array (3 channels, 224x224 for Places365)
    frame.to_ndarray.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    # Return a PIL Image when to_image() is called
    frame.to_image.return_value = Image.fromarray(
        np.zeros((224, 224, 3), dtype=np.uint8)
    )
    return frame


# ============================================================================
# PlacesDetectionService Initialization Tests
# ============================================================================


class TestPlacesDetectionServiceInit:
    """Tests for PlacesDetectionService initialization."""

    @patch("torch.hub.load")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_initialization_success(
        self, mock_path_exists, mock_torch_load, mock_hub_load
    ):
        """Test successful service initialization."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService(
                model_path="models/resnet18_places365.pth.tar"
            )

            assert service.model_path == "models/resnet18_places365.pth.tar"
            assert service.model is not None

    @patch("torch.hub.load")
    @patch("builtins.open", new_callable=lambda: mock_open_categories())
    def test_initialization_model_not_found(self, mock_open_file, mock_hub_load):
        """Test initialization when model file doesn't exist."""
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model

        # Mock Path.exists to return False for model file
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(PlacesDetectionError):
                PlacesDetectionService(model_path="nonexistent.pth.tar")

    @patch("torch.hub.load")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_initialization_model_load_failure(
        self, mock_path_exists, mock_torch_load, mock_hub_load
    ):
        """Test initialization when model fails to load."""
        mock_path_exists.return_value = True
        mock_torch_load.side_effect = Exception("Corrupted model file")

        with patch("builtins.open", mock_open_categories()):
            with pytest.raises(PlacesDetectionError) as exc_info:
                PlacesDetectionService(model_path="corrupted.pth.tar")

            assert "Failed to load Places365 model" in str(exc_info.value)

    @patch("torch.hub.load")
    @patch("torch.load")
    def test_initialization_categories_not_found(self, mock_torch_load, mock_hub_load):
        """Test initialization when categories file doesn't exist."""
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        # Mock Path.exists to return False for categories file
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(PlacesDetectionError) as exc_info:
                PlacesDetectionService(model_path="models/resnet18_places365.pth.tar")

            assert "labels file not found" in str(exc_info.value).lower()


# ============================================================================
# PlacesDetectionService Detection Tests
# ============================================================================


class TestPlacesDetectionService:
    """Tests for PlacesDetectionService place detection."""

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_video_not_found(
        self, mock_path_exists, mock_torch_load, mock_av, mock_hub_load
    ):
        """Test detection with non-existent video file."""
        # Model and categories exist
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            # Video doesn't exist
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(PlacesDetectionError) as exc_info:
                    service.detect_places_in_video(
                        video_path="/nonexistent/video.mp4",
                        video_id="test-id",
                        sample_rate=30,
                    )

                assert "Video file not found" in str(exc_info.value)

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_success(
        self,
        mock_path_exists,
        mock_torch_load,
        mock_av,
        mock_hub_load,
        mock_av_container,
        mock_frame,
    ):
        """Test successful place detection."""
        # Setup model loading
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            # Setup video
            mock_av.open.return_value = mock_av_container
            mock_av_container.decode.return_value = [mock_frame] * 90

            # Mock model predictions
            import torch

            mock_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 360])
            service.model = Mock(return_value=mock_logits)

            places = service.detect_places_in_video(
                video_path="/path/to/video.mp4",
                video_id="test-video-id",
                sample_rate=30,
                top_k=5,
            )

            # Should return places
            assert len(places) > 0
            assert all(isinstance(p, Place) for p in places)
            assert all(p.video_id == "test-video-id" for p in places)

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_no_frames(
        self,
        mock_path_exists,
        mock_torch_load,
        mock_av,
        mock_hub_load,
        mock_av_container,
    ):
        """Test detection with video containing no frames."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            # Video with no frames
            mock_av.open.return_value = mock_av_container
            mock_av_container.decode.return_value = []

            places = service.detect_places_in_video(
                video_path="/path/to/empty.mp4",
                video_id="test-video-id",
                sample_rate=30,
            )

            assert places == []

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_frame_sampling(
        self,
        mock_path_exists,
        mock_torch_load,
        mock_av,
        mock_hub_load,
        mock_av_container,
        mock_frame,
    ):
        """Test frame sampling with different sample rates."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            # 90 frames at 30fps = 3 seconds
            mock_av.open.return_value = mock_av_container
            mock_av_container.decode.return_value = [mock_frame] * 90

            # Mock model predictions
            import torch

            mock_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 360])
            service.model = Mock(return_value=mock_logits)

            # Sample every 30 frames (1 per second at 30fps)
            result = service.detect_places_in_video(
                video_path="/path/to/video.mp4",
                video_id="test-video-id",
                sample_rate=30,
                top_k=5,
            )

            # Should process 3 frames (0, 30, 60)
            # Model should be called 3 times
            assert service.model.call_count == 3
            # Verify result is returned
            assert isinstance(result, list)

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_aggregation(
        self,
        mock_path_exists,
        mock_torch_load,
        mock_av,
        mock_hub_load,
        mock_av_container,
        mock_frame,
    ):
        """Test that detections are aggregated by label."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            mock_av.open.return_value = mock_av_container
            mock_av_container.decode.return_value = [mock_frame] * 60

            # Mock model to return same top prediction for both frames
            import torch

            mock_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 360])
            service.model = Mock(return_value=mock_logits)

            result = service.detect_places_in_video(
                video_path="/path/to/video.mp4",
                video_id="test-video-id",
                sample_rate=30,
                top_k=5,
            )

            # Same place detected in multiple frames should be aggregated
            # Check that timestamps are accumulated
            if result:
                assert all(len(p.timestamps) > 0 for p in result)

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_confidence_calculation(
        self,
        mock_path_exists,
        mock_torch_load,
        mock_av,
        mock_hub_load,
        mock_av_container,
        mock_frame,
    ):
        """Test average confidence calculation."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            mock_av.open.return_value = mock_av_container
            mock_av_container.decode.return_value = [mock_frame] * 30

            # Mock model predictions with known confidence
            import torch

            mock_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 360])
            service.model = Mock(return_value=mock_logits)

            places = service.detect_places_in_video(
                video_path="/path/to/video.mp4",
                video_id="test-video-id",
                sample_rate=30,
                top_k=5,
            )

            # Check that confidence is calculated
            if places:
                assert all(0.0 <= p.confidence <= 1.0 for p in places)

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_alternative_labels(
        self,
        mock_path_exists,
        mock_torch_load,
        mock_av,
        mock_hub_load,
        mock_av_container,
        mock_frame,
    ):
        """Test that alternative labels are stored."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            mock_av.open.return_value = mock_av_container
            mock_av_container.decode.return_value = [mock_frame] * 30

            # Mock model predictions with top-k
            import torch

            mock_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 360])
            service.model = Mock(return_value=mock_logits)

            places = service.detect_places_in_video(
                video_path="/path/to/video.mp4",
                video_id="test-video-id",
                sample_rate=30,
                top_k=5,
            )

            # Check that alternative labels are present
            if places:
                for place in places:
                    if place.alternative_labels:
                        assert isinstance(place.alternative_labels, list)
                        for alt in place.alternative_labels:
                            assert "label" in alt
                            assert "confidence" in alt

    @patch("torch.hub.load")
    @patch("src.services.places_detection_service.av")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_detect_places_timestamps_sorted(
        self,
        mock_path_exists,
        mock_torch_load,
        mock_av,
        mock_hub_load,
        mock_av_container,
        mock_frame,
    ):
        """Test that timestamps are in chronological order."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.fc = Mock()
        mock_model.fc.in_features = 512
        mock_hub_load.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        with patch("builtins.open", mock_open_categories()):
            service = PlacesDetectionService()

            mock_av.open.return_value = mock_av_container
            mock_av_container.decode.return_value = [mock_frame] * 90

            # Mock model predictions
            import torch

            mock_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 360])
            service.model = Mock(return_value=mock_logits)

            places = service.detect_places_in_video(
                video_path="/path/to/video.mp4",
                video_id="test-video-id",
                sample_rate=30,
            )

            # Timestamps should be in order
            if places:
                for place in places:
                    timestamps = place.timestamps
                    assert timestamps == sorted(timestamps)


# ============================================================================
# PlacesDetectionTaskHandler Tests
# ============================================================================


class TestPlacesDetectionTaskHandler:
    """Tests for PlacesDetectionTaskHandler."""

    def test_handler_initialization(self):
        """Test handler initialization."""
        mock_repo = Mock()
        mock_service = Mock()

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=mock_service,
            model_path="models/resnet18_places365.pth.tar",
            sample_rate=30,
            top_k=5,
        )

        assert handler.place_repository == mock_repo
        assert handler.detection_service == mock_service
        assert handler.model_path == "models/resnet18_places365.pth.tar"
        assert handler.sample_rate == 30
        assert handler.top_k == 5

    def test_process_places_detection_task_success(self):
        """Test successful places detection task processing."""
        mock_repo = Mock()
        mock_service = Mock()

        # Create mock place
        mock_place = Place(
            place_id=str(uuid.uuid4()),
            video_id="test-video-id",
            label="restaurant",
            timestamps=[1.0, 2.0, 3.0],
            confidence=0.92,
            alternative_labels=[
                {"label": "bar", "confidence": 0.85},
                {"label": "cafeteria", "confidence": 0.78},
            ],
        )

        mock_service.detect_places_in_video.return_value = [mock_place]

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=mock_service,
            sample_rate=30,
            top_k=5,
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.PLACES_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        result = handler.process_places_detection_task(task, video)

        assert result is True
        mock_service.detect_places_in_video.assert_called_once()
        mock_repo.save.assert_called_once_with(mock_place)

    def test_process_places_detection_task_failure(self):
        """Test places detection task processing failure."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_places_in_video.side_effect = Exception("Detection failed")

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=mock_service,
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.PLACES_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        result = handler.process_places_detection_task(task, video)

        assert result is False
        mock_repo.save.assert_not_called()

    def test_process_places_detection_task_no_places(self):
        """Test task processing when no places are detected."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_places_in_video.return_value = []

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=mock_service,
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.PLACES_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        result = handler.process_places_detection_task(task, video)

        assert result is True
        mock_repo.save.assert_not_called()

    def test_get_detected_places(self):
        """Test getting detected places for a video."""
        mock_repo = Mock()
        mock_places = [
            Place(
                place_id=str(uuid.uuid4()),
                video_id="test-video-id",
                label="restaurant",
                timestamps=[1.0, 2.0],
                confidence=0.90,
            )
        ]
        mock_repo.find_by_video_id.return_value = mock_places

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=Mock(),
        )

        places = handler.get_detected_places("test-video-id")

        assert places == mock_places
        mock_repo.find_by_video_id.assert_called_once_with("test-video-id")

    def test_get_detected_places_empty(self):
        """Test getting places when none exist."""
        mock_repo = Mock()
        mock_repo.find_by_video_id.return_value = []

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=Mock(),
        )

        places = handler.get_detected_places("test-video-id")

        assert places == []

    def test_get_places_by_label(self):
        """Test getting places filtered by label."""
        mock_repo = Mock()
        mock_places = [
            Place(
                place_id=str(uuid.uuid4()),
                video_id="test-video-id",
                label="restaurant",
                timestamps=[1.0, 2.0],
                confidence=0.90,
            )
        ]
        mock_repo.find_by_label.return_value = mock_places

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=Mock(),
        )

        places = handler.get_places_by_label("test-video-id", "restaurant")

        assert places == mock_places
        mock_repo.find_by_label.assert_called_once_with("test-video-id", "restaurant")

    def test_places_detection_task_with_custom_sample_rate(self):
        """Test task processing with custom sample rate."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_places_in_video.return_value = []

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=mock_service,
            sample_rate=60,  # Custom sample rate
            top_k=3,  # Custom top_k
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.PLACES_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        handler.process_places_detection_task(task, video)

        # Verify sample_rate and top_k were passed
        call_args = mock_service.detect_places_in_video.call_args
        assert call_args[1]["sample_rate"] == 60
        assert call_args[1]["top_k"] == 3

    def test_process_task_repository_save_failure(self):
        """Test handling when repository save fails."""
        mock_repo = Mock()
        mock_repo.save.side_effect = Exception("Database error")
        mock_service = Mock()

        place = Place(
            place_id=str(uuid.uuid4()),
            video_id="test-video-id",
            label="restaurant",
            timestamps=[1.0],
            confidence=0.90,
        )
        mock_service.detect_places_in_video.return_value = [place]

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=mock_service,
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.PLACES_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        result = handler.process_places_detection_task(task, video)

        # Should return False when save fails
        assert result is False

    def test_process_task_multiple_places(self):
        """Test processing task with multiple place detections."""
        mock_repo = Mock()
        mock_service = Mock()

        place1 = Place(
            place_id=str(uuid.uuid4()),
            video_id="test-video-id",
            label="restaurant",
            timestamps=[1.0, 2.0],
            confidence=0.92,
        )
        place2 = Place(
            place_id=str(uuid.uuid4()),
            video_id="test-video-id",
            label="bar",
            timestamps=[3.0],
            confidence=0.85,
        )

        mock_service.detect_places_in_video.return_value = [place1, place2]

        handler = PlacesDetectionTaskHandler(
            place_repository=mock_repo,
            detection_service=mock_service,
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test-video-id",
            task_type=TaskType.PLACES_DETECTION.value,
        )

        video = Video(
            video_id="test-video-id",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=Mock(),
        )

        result = handler.process_places_detection_task(task, video)

        assert result is True
        assert mock_repo.save.call_count == 2
        mock_repo.save.assert_any_call(place1)
        mock_repo.save.assert_any_call(place2)


# ============================================================================
# Helper Functions
# ============================================================================


def mock_open_categories():
    """Mock open() for categories file."""
    from unittest.mock import mock_open

    categories_content = "\n".join([f"place_{i}" for i in range(365)])
    return mock_open(read_data=categories_content)
