"""Tests for object detection service and task handler."""

import uuid
from unittest.mock import Mock, patch

import pytest

from src.domain.models import Object, Task, Video
from src.services.object_detection_service import (
    ObjectDetectionError,
    ObjectDetectionService,
)
from src.services.object_detection_task_handler import ObjectDetectionTaskHandler
from src.services.task_orchestration import TaskType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    model = Mock()
    model.names = {0: "person", 1: "car", 2: "dog", 3: "cat"}
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
def mock_detection_box():
    """Create a mock YOLO detection box."""
    box = Mock()
    box.cls = [0]  # person class
    box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
    box.conf = [0.95]
    return box


# ============================================================================
# ObjectDetectionService Initialization Tests
# ============================================================================


class TestObjectDetectionServiceInit:
    """Tests for ObjectDetectionService initialization."""

    @patch("ultralytics.YOLO")
    def test_initialization_success(self, mock_yolo_class):
        """Test successful service initialization."""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model

        service = ObjectDetectionService(model_name="yolov8n.pt")

        assert service.model_name == "yolov8n.pt"
        assert service.model == mock_model
        mock_yolo_class.assert_called_once_with("yolov8n.pt")

    @patch("ultralytics.YOLO")
    def test_initialization_with_different_models(self, mock_yolo_class):
        """Test initialization with different model variants."""
        model_variants = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

        for model_name in model_variants:
            mock_yolo_class.reset_mock()
            service = ObjectDetectionService(model_name=model_name)
            assert service.model_name == model_name
            mock_yolo_class.assert_called_with(model_name)

    @patch("ultralytics.YOLO")
    def test_initialization_model_load_failure(self, mock_yolo_class):
        """Test initialization when model fails to load."""
        mock_yolo_class.side_effect = Exception("Model file not found")

        with pytest.raises(ObjectDetectionError) as exc_info:
            ObjectDetectionService(model_name="nonexistent.pt")

        assert "Failed to load YOLO model" in str(exc_info.value)

    def test_initialization_ultralytics_not_installed(self):
        """Test initialization when ultralytics is not installed."""
        with patch.dict("sys.modules", {"ultralytics": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'ultralytics'"),
            ):
                with pytest.raises(ObjectDetectionError) as exc_info:
                    ObjectDetectionService()

                assert "ultralytics package not installed" in str(exc_info.value)


# ============================================================================
# ObjectDetectionService Video Detection Tests
# ============================================================================


class TestObjectDetectionServiceDetection:
    """Tests for ObjectDetectionService detection functionality."""

    @patch("ultralytics.YOLO")
    def test_detect_objects_nonexistent_video(self, mock_yolo_class):
        """Test detection with nonexistent video file."""
        service = ObjectDetectionService()

        with pytest.raises(ObjectDetectionError) as exc_info:
            service.detect_objects_in_video(
                video_path="/nonexistent/path/video.mp4",
                video_id="test-video-id",
            )

        assert "Video file not found" in str(exc_info.value)

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_invalid_video_format(
        self, mock_path, mock_yolo_class, mock_av
    ):
        """Test detection with invalid video format."""
        mock_path.return_value.exists.return_value = True

        # Create a real exception class for av.AVError
        class MockAVError(Exception):
            pass

        mock_av.AVError = MockAVError
        mock_av.open.side_effect = MockAVError("Invalid video format")

        service = ObjectDetectionService()

        with pytest.raises(ObjectDetectionError) as exc_info:
            service.detect_objects_in_video(
                video_path="/path/to/invalid.mp4",
                video_id="test-video-id",
            )

        assert "Failed to open video" in str(exc_info.value)

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_empty_video(self, mock_path, mock_yolo_class, mock_av):
        """Test detection with video containing no frames."""
        mock_path.return_value.exists.return_value = True

        # Mock container with no frames
        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 0
        stream.codec_context.name = "h264"
        container.streams.video = [stream]
        container.decode.return_value = []  # No frames
        mock_av.open.return_value = container

        service = ObjectDetectionService()
        service.model = Mock()
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/empty.mp4",
            video_id="test-video-id",
        )

        assert objects == []
        container.close.assert_called_once()

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_no_detections(self, mock_path, mock_yolo_class, mock_av):
        """Test detection when YOLO finds no objects."""
        mock_path.return_value.exists.return_value = True

        # Mock container with frames
        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 90
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame, frame, frame]
        mock_av.open.return_value = container

        # Mock model returning no detections
        service = ObjectDetectionService()
        mock_result = Mock()
        mock_result.boxes = []  # No detections
        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="test-video-id",
            sample_rate=1,
        )

        assert objects == []

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_single_label(self, mock_path, mock_yolo_class, mock_av):
        """Test detection with single object label across frames."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 60
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame1 = Mock()
        frame1.to_ndarray.return_value = Mock()
        frame2 = Mock()
        frame2.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame1, frame2]
        mock_av.open.return_value = container

        # Mock YOLO detecting "person" in both frames
        service = ObjectDetectionService()
        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.92]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-123",
            sample_rate=1,
        )

        assert len(objects) == 1
        assert objects[0].label == "person"
        assert objects[0].video_id == "vid-123"
        assert len(objects[0].timestamps) == 2
        assert len(objects[0].bounding_boxes) == 2

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_multiple_labels(self, mock_path, mock_yolo_class, mock_av):
        """Test detection with multiple object labels."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        # Mock YOLO detecting person and car in same frame
        service = ObjectDetectionService()

        person_box = Mock()
        person_box.cls = [0]
        person_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        person_box.conf = [0.95]

        car_box = Mock()
        car_box.cls = [1]
        car_box.xyxy = [Mock(tolist=Mock(return_value=[150.0, 100.0, 400.0, 300.0]))]
        car_box.conf = [0.88]

        mock_result = Mock()
        mock_result.boxes = [person_box, car_box]

        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person", 1: "car"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-456",
            sample_rate=1,
        )

        assert len(objects) == 2
        labels = {obj.label for obj in objects}
        assert labels == {"person", "car"}

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_sample_rate(self, mock_path, mock_yolo_class, mock_av):
        """Test that sample_rate controls frame processing frequency."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 90
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        # Create 90 mock frames
        frames = [Mock() for _ in range(90)]
        for f in frames:
            f.to_ndarray.return_value = Mock()
        container.decode.return_value = frames
        mock_av.open.return_value = container

        service = ObjectDetectionService()
        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.9]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        # With sample_rate=30, should process frames 0, 30, 60 (3 frames)
        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-789",
            sample_rate=30,
        )

        assert len(objects) == 1
        assert len(objects[0].timestamps) == 3  # 90 frames / 30 sample_rate = 3

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_zero_fps_fallback(
        self, mock_path, mock_yolo_class, mock_av
    ):
        """Test timestamp calculation when fps is zero."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 0.0  # Zero FPS
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        service = ObjectDetectionService()
        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.9]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-zero-fps",
            sample_rate=1,
        )

        # When fps is 0, timestamp should fallback to frame_idx
        assert len(objects) == 1
        assert objects[0].timestamps[0] == 0  # frame_idx as timestamp

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_bounding_box_format(
        self, mock_path, mock_yolo_class, mock_av
    ):
        """Test that bounding boxes contain expected fields."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        service = ObjectDetectionService()
        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.5, 20.5, 100.5, 200.5]))]
        mock_box.conf = [0.87]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-bbox",
            sample_rate=1,
        )

        bbox = objects[0].bounding_boxes[0]
        assert "frame" in bbox
        assert "timestamp" in bbox
        assert "bbox" in bbox
        assert "confidence" in bbox
        assert bbox["frame"] == 0
        assert bbox["timestamp"] == 0.0
        assert bbox["bbox"] == [10.5, 20.5, 100.5, 200.5]
        assert bbox["confidence"] == 0.87

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_detect_objects_timestamps_calculated_correctly(
        self, mock_path, mock_yolo_class, mock_av
    ):
        """Test that timestamps are calculated correctly from frame index and fps."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0  # 30 fps
        stream.frames = 90
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        # Create frames at indices 0, 30, 60
        frames = [Mock() for _ in range(90)]
        for f in frames:
            f.to_ndarray.return_value = Mock()
        container.decode.return_value = frames
        mock_av.open.return_value = container

        service = ObjectDetectionService()
        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.9]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-timestamps",
            sample_rate=30,
        )

        # At 30 fps, frames 0, 30, 60 should have timestamps 0.0, 1.0, 2.0
        assert objects[0].timestamps == [0.0, 1.0, 2.0]


# ============================================================================
# Frame Processing Edge Cases
# ============================================================================


class TestFrameProcessingEdgeCases:
    """Tests for frame processing edge cases."""

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_frame_processing_error_continues(
        self, mock_path, mock_yolo_class, mock_av
    ):
        """Test that frame processing errors don't stop detection."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 60
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame1 = Mock()
        frame1.to_ndarray.return_value = Mock()
        frame2 = Mock()
        frame2.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame1, frame2]
        mock_av.open.return_value = container

        service = ObjectDetectionService()

        # First call raises error, second succeeds
        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.9]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        call_count = [0]

        def model_side_effect(frame, verbose=False):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("GPU memory error")
            return [mock_result]

        service.model = Mock(side_effect=model_side_effect)
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-error",
            sample_rate=1,
        )

        # Should still get results from second frame
        assert len(objects) == 1
        assert len(objects[0].timestamps) == 1

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_container_closed_on_exception(self, mock_path, mock_yolo_class, mock_av):
        """Test that video container is closed even when exception occurs."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        # Raise exception during decode
        container.decode.side_effect = RuntimeError("Decode error")
        mock_av.open.return_value = container

        service = ObjectDetectionService()

        # The service catches exceptions during frame processing but not decode
        # so we need to verify container.close is called in finally block
        try:
            service.detect_objects_in_video(
                video_path="/path/to/video.mp4",
                video_id="vid-exception",
                sample_rate=1,
            )
        except RuntimeError:
            pass

        container.close.assert_called_once()

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_multiple_detections_same_class_same_frame(
        self, mock_path, mock_yolo_class, mock_av
    ):
        """Test multiple detections of same class in one frame."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        # Two people in same frame
        person1_box = Mock()
        person1_box.cls = [0]
        person1_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        person1_box.conf = [0.95]

        person2_box = Mock()
        person2_box.cls = [0]
        person2_box.xyxy = [Mock(tolist=Mock(return_value=[200.0, 50.0, 350.0, 250.0]))]
        person2_box.conf = [0.88]

        mock_result = Mock()
        mock_result.boxes = [person1_box, person2_box]

        service = ObjectDetectionService()
        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-multi",
            sample_rate=1,
        )

        # Should aggregate both detections under "person"
        assert len(objects) == 1
        assert objects[0].label == "person"
        assert len(objects[0].bounding_boxes) == 2

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_low_confidence_detections_stored(
        self, mock_path, mock_yolo_class, mock_av
    ):
        """Test that low confidence detections are still stored (no filtering)."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        # Very low confidence detection
        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.15]  # Very low confidence

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service = ObjectDetectionService()
        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-lowconf",
            sample_rate=1,
        )

        # Currently no filtering - low confidence detections are stored
        assert len(objects) == 1
        assert objects[0].bounding_boxes[0]["confidence"] == 0.15

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_many_object_classes_aggregation(self, mock_path, mock_yolo_class, mock_av):
        """Test aggregation with many different object classes."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        # Create boxes for many different classes
        boxes = []
        class_names = {i: f"class_{i}" for i in range(10)}
        for i in range(10):
            box = Mock()
            box.cls = [i]
            box.xyxy = [
                Mock(tolist=Mock(return_value=[i * 10.0, i * 10.0, i * 20.0, i * 20.0]))
            ]
            box.conf = [0.8 + i * 0.01]
            boxes.append(box)

        mock_result = Mock()
        mock_result.boxes = boxes

        service = ObjectDetectionService()
        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = class_names

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-many-classes",
            sample_rate=1,
        )

        assert len(objects) == 10
        labels = {obj.label for obj in objects}
        expected_labels = {f"class_{i}" for i in range(10)}
        assert labels == expected_labels


# ============================================================================
# Object Domain Model Tests
# ============================================================================


class TestObjectDomainModel:
    """Tests for Object domain model integration."""

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_object_has_valid_uuid(self, mock_path, mock_yolo_class, mock_av):
        """Test that returned objects have valid UUIDs."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.9]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service = ObjectDetectionService()
        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id="vid-uuid",
            sample_rate=1,
        )

        # Verify UUID is valid
        obj_uuid = uuid.UUID(objects[0].object_id)
        assert obj_uuid.version == 4

    @patch("src.services.object_detection_service.av")
    @patch("ultralytics.YOLO")
    @patch("src.services.object_detection_service.Path")
    def test_object_video_id_preserved(self, mock_path, mock_yolo_class, mock_av):
        """Test that video_id is correctly associated with objects."""
        mock_path.return_value.exists.return_value = True

        container = Mock()
        stream = Mock()
        stream.average_rate = 30.0
        stream.frames = 30
        stream.codec_context.name = "h264"
        container.streams.video = [stream]

        frame = Mock()
        frame.to_ndarray.return_value = Mock()
        container.decode.return_value = [frame]
        mock_av.open.return_value = container

        mock_box = Mock()
        mock_box.cls = [0]
        mock_box.xyxy = [Mock(tolist=Mock(return_value=[10.0, 20.0, 100.0, 200.0]))]
        mock_box.conf = [0.9]

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        service = ObjectDetectionService()
        service.model = Mock()
        service.model.return_value = [mock_result]
        service.model.names = {0: "person"}

        expected_video_id = "my-unique-video-id-12345"
        objects = service.detect_objects_in_video(
            video_path="/path/to/video.mp4",
            video_id=expected_video_id,
            sample_rate=1,
        )

        assert objects[0].video_id == expected_video_id


# ============================================================================
# ObjectDetectionTaskHandler Tests
# ============================================================================


class TestObjectDetectionTaskHandler:
    """Tests for ObjectDetectionTaskHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        mock_repo = Mock()
        handler = ObjectDetectionTaskHandler(object_repository=mock_repo)
        assert handler.object_repository == mock_repo
        assert handler.detection_service is not None

    def test_initialization_with_custom_service(self):
        """Test handler initialization with custom detection service."""
        mock_repo = Mock()
        mock_service = Mock()
        handler = ObjectDetectionTaskHandler(
            object_repository=mock_repo, detection_service=mock_service
        )
        assert handler.detection_service == mock_service

    def test_process_object_detection_task_success(self):
        """Test successful object detection task processing."""
        mock_repo = Mock()
        mock_service = Mock()

        obj1 = Object(
            object_id=str(uuid.uuid4()),
            video_id="test_video",
            label="person",
            timestamps=[1.0, 2.0],
            bounding_boxes=[
                {
                    "frame": 30,
                    "timestamp": 1.0,
                    "bbox": [10, 20, 30, 40],
                    "confidence": 0.9,
                },
                {
                    "frame": 60,
                    "timestamp": 2.0,
                    "bbox": [15, 25, 35, 45],
                    "confidence": 0.85,
                },
            ],
        )
        obj2 = Object(
            object_id=str(uuid.uuid4()),
            video_id="test_video",
            label="car",
            timestamps=[1.5],
            bounding_boxes=[
                {
                    "frame": 45,
                    "timestamp": 1.5,
                    "bbox": [100, 200, 300, 400],
                    "confidence": 0.95,
                }
            ],
        )

        mock_service.detect_objects_in_video.return_value = [obj1, obj2]

        handler = ObjectDetectionTaskHandler(
            object_repository=mock_repo, detection_service=mock_service
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test_video",
            task_type=TaskType.OBJECT_DETECTION.value,
            status="pending",
        )
        video = Video(
            video_id="test_video",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=1234567890.0,
        )

        result = handler.process_object_detection_task(task, video)

        assert result is True
        mock_service.detect_objects_in_video.assert_called_once_with(
            video_path="/path/to/video.mp4", video_id="test_video", sample_rate=30
        )
        assert mock_repo.save.call_count == 2
        mock_repo.save.assert_any_call(obj1)
        mock_repo.save.assert_any_call(obj2)

    def test_process_object_detection_task_failure(self):
        """Test object detection task processing with failure."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_objects_in_video.side_effect = Exception("Detection failed")

        handler = ObjectDetectionTaskHandler(
            object_repository=mock_repo, detection_service=mock_service
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test_video",
            task_type=TaskType.OBJECT_DETECTION.value,
            status="pending",
        )
        video = Video(
            video_id="test_video",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=1234567890.0,
        )

        result = handler.process_object_detection_task(task, video)

        assert result is False
        mock_repo.save.assert_not_called()

    def test_process_object_detection_task_no_objects(self):
        """Test object detection task when no objects are detected."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_objects_in_video.return_value = []

        handler = ObjectDetectionTaskHandler(
            object_repository=mock_repo, detection_service=mock_service
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test_video",
            task_type=TaskType.OBJECT_DETECTION.value,
            status="pending",
        )
        video = Video(
            video_id="test_video",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=1234567890.0,
        )

        result = handler.process_object_detection_task(task, video)

        assert result is True
        mock_repo.save.assert_not_called()

    def test_get_detected_objects(self):
        """Test retrieving detected objects for a video."""
        mock_repo = Mock()
        mock_objects = [Mock(), Mock()]
        mock_repo.find_by_video_id.return_value = mock_objects

        handler = ObjectDetectionTaskHandler(object_repository=mock_repo)
        result = handler.get_detected_objects("test_video")

        assert result == mock_objects
        mock_repo.find_by_video_id.assert_called_once_with("test_video")

    def test_get_objects_by_label(self):
        """Test retrieving detected objects filtered by label."""
        mock_repo = Mock()
        mock_objects = [Mock()]
        mock_repo.find_by_label.return_value = mock_objects

        handler = ObjectDetectionTaskHandler(object_repository=mock_repo)
        result = handler.get_objects_by_label("test_video", "person")

        assert result == mock_objects
        mock_repo.find_by_label.assert_called_once_with("test_video", "person")

    def test_process_task_repository_save_failure(self):
        """Test handling when repository save fails."""
        mock_repo = Mock()
        mock_repo.save.side_effect = Exception("Database error")
        mock_service = Mock()

        obj = Object(
            object_id=str(uuid.uuid4()),
            video_id="test_video",
            label="person",
            timestamps=[1.0],
            bounding_boxes=[
                {
                    "frame": 30,
                    "timestamp": 1.0,
                    "bbox": [10, 20, 30, 40],
                    "confidence": 0.9,
                }
            ],
        )
        mock_service.detect_objects_in_video.return_value = [obj]

        handler = ObjectDetectionTaskHandler(
            object_repository=mock_repo, detection_service=mock_service
        )

        task = Task(
            task_id=str(uuid.uuid4()),
            video_id="test_video",
            task_type=TaskType.OBJECT_DETECTION.value,
            status="pending",
        )
        video = Video(
            video_id="test_video",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=1234567890.0,
        )

        result = handler.process_object_detection_task(task, video)

        # Should return False when save fails
        assert result is False

    def test_process_task_with_special_characters_in_video_id(self):
        """Test handling video IDs with special characters."""
        mock_repo = Mock()
        mock_service = Mock()
        mock_service.detect_objects_in_video.return_value = []

        handler = ObjectDetectionTaskHandler(
            object_repository=mock_repo, detection_service=mock_service
        )

        special_video_id = "video-with-special_chars.123"
        task = Task(
            task_id=str(uuid.uuid4()),
            video_id=special_video_id,
            task_type=TaskType.OBJECT_DETECTION.value,
            status="pending",
        )
        video = Video(
            video_id=special_video_id,
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=1234567890.0,
        )

        result = handler.process_object_detection_task(task, video)

        assert result is True
        mock_service.detect_objects_in_video.assert_called_once_with(
            video_path="/path/to/video.mp4",
            video_id=special_video_id,
            sample_rate=30,
        )
