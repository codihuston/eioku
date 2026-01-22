"""Tests for face detection task handler."""

import json
import uuid
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.domain.artifacts import ArtifactEnvelope
from src.domain.models import Task, Video
from src.domain.schema_registry import SchemaRegistry
from src.services.face_detection_task_handler import FaceDetectionTaskHandler


class TestFaceDetectionTaskHandler:
    """Test suite for FaceDetectionTaskHandler."""

    @pytest.fixture
    def mock_artifact_repository(self):
        """Create mock artifact repository."""
        repo = Mock()
        repo.create = Mock(return_value=None)
        repo.get_by_asset = Mock(return_value=[])
        return repo

    @pytest.fixture
    def mock_detection_service(self):
        """Create mock face detection service."""
        service = Mock()
        
        # Mock service returns frame-level detections
        service.detect_faces_in_video = Mock(return_value=[
            {
                "frame_number": 30,
                "timestamp": 1.0,
                "detections": [
                    {
                        "bbox": [100.0, 150.0, 200.0, 250.0],
                        "confidence": 0.95,
                    }
                ]
            },
            {
                "frame_number": 60,
                "timestamp": 2.0,
                "detections": [
                    {
                        "bbox": [110.0, 160.0, 210.0, 260.0],
                        "confidence": 0.92,
                    }
                ]
            },
        ])
        return service

    @pytest.fixture
    def handler(self, mock_artifact_repository, mock_detection_service):
        """Create face detection task handler."""
        return FaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_detection_service,
            model_name="yolov8n-face.pt",
            sample_rate=30,
        )

    def test_process_face_detection_task_success(
        self, handler, mock_artifact_repository, mock_detection_service
    ):
        """Test successful face detection processing."""
        # Arrange
        task = Task(
            task_id="task-1",
            video_id="video-123",
            task_type="face_detection",
            status="pending",
        )
        video = Video(
            video_id="video-123",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=datetime.utcnow(),
        )
        run_id = str(uuid.uuid4())

        # Act
        result = handler.process_face_detection_task(task, video, run_id)

        # Assert
        assert result is True
        mock_detection_service.detect_faces_in_video.assert_called_once_with(
            video_path="/path/to/video.mp4",
            sample_rate=30,
        )
        # Should create 2 artifacts (one per detection)
        assert mock_artifact_repository.create.call_count == 2

        # Verify first artifact
        first_call = mock_artifact_repository.create.call_args_list[0]
        artifact = first_call[0][0]
        assert artifact.artifact_type == "face.detection"
        assert artifact.asset_id == "video-123"
        assert artifact.schema_version == 1
        assert artifact.producer == "yolo-face"
        assert artifact.model_profile == "fast"
        assert artifact.run_id == run_id

        # Verify payload structure
        payload = json.loads(artifact.payload_json)
        assert "confidence" in payload
        assert "bounding_box" in payload
        assert "cluster_id" in payload
        assert "frame_number" in payload
        # Cluster ID is auto-generated now, just verify it exists
        assert payload["cluster_id"].startswith("face_")

    def test_process_face_detection_task_generates_run_id(
        self, handler, mock_artifact_repository, mock_detection_service
    ):
        """Test that run_id is generated if not provided."""
        # Arrange
        task = Task(
            task_id="task-1",
            video_id="video-123",
            task_type="face_detection",
            status="pending",
        )
        video = Video(
            video_id="video-123",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=datetime.utcnow(),
        )

        # Act
        result = handler.process_face_detection_task(task, video, run_id=None)

        # Assert
        assert result is True
        # Verify that artifacts were created with a run_id
        assert mock_artifact_repository.create.call_count == 2
        first_call = mock_artifact_repository.create.call_args_list[0]
        artifact = first_call[0][0]
        assert artifact.run_id is not None
        assert len(artifact.run_id) > 0

    def test_process_face_detection_task_failure(
        self, handler, mock_artifact_repository, mock_detection_service
    ):
        """Test face detection processing failure."""
        # Arrange
        task = Task(
            task_id="task-1",
            video_id="video-123",
            task_type="face_detection",
            status="pending",
        )
        video = Video(
            video_id="video-123",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=datetime.utcnow(),
        )
        mock_detection_service.detect_faces_in_video.side_effect = Exception(
            "Detection failed"
        )

        # Act
        result = handler.process_face_detection_task(task, video)

        # Assert
        assert result is False
        mock_artifact_repository.create.assert_not_called()

    def test_get_detected_faces(self, handler, mock_artifact_repository):
        """Test getting detected faces for a video."""
        # Arrange
        video_id = "video-123"
        mock_artifacts = [
            ArtifactEnvelope(
                artifact_id="artifact-1",
                asset_id=video_id,
                artifact_type="face.detection",
                schema_version=1,
                span_start_ms=1000,
                span_end_ms=1001,
                payload_json='{"confidence": 0.95}',
                producer="yolo-face",
                producer_version="yolov8n-face.pt",
                model_profile="fast",
                config_hash="abc123",
                input_hash="def456",
                run_id="run-1",
                created_at=datetime.utcnow(),
            )
        ]
        mock_artifact_repository.get_by_asset.return_value = mock_artifacts

        # Act
        result = handler.get_detected_faces(video_id)

        # Assert
        assert result == mock_artifacts
        mock_artifact_repository.get_by_asset.assert_called_once_with(
            asset_id=video_id, artifact_type="face.detection"
        )

    def test_get_faces_by_cluster(self, handler, mock_artifact_repository):
        """Test getting faces filtered by cluster ID."""
        # Arrange
        video_id = "video-123"
        cluster_id = "person_001"
        mock_artifacts = [
            ArtifactEnvelope(
                artifact_id="artifact-1",
                asset_id=video_id,
                artifact_type="face.detection",
                schema_version=1,
                span_start_ms=1000,
                span_end_ms=1001,
                payload_json='{"confidence": 0.95, "cluster_id": "person_001"}',
                producer="yolo-face",
                producer_version="yolov8n-face.pt",
                model_profile="fast",
                config_hash="abc123",
                input_hash="def456",
                run_id="run-1",
                created_at=datetime.utcnow(),
            ),
            ArtifactEnvelope(
                artifact_id="artifact-2",
                asset_id=video_id,
                artifact_type="face.detection",
                schema_version=1,
                span_start_ms=2000,
                span_end_ms=2001,
                payload_json='{"confidence": 0.92, "cluster_id": "person_002"}',
                producer="yolo-face",
                producer_version="yolov8n-face.pt",
                model_profile="fast",
                config_hash="abc123",
                input_hash="def456",
                run_id="run-1",
                created_at=datetime.utcnow(),
            ),
        ]
        mock_artifact_repository.get_by_asset.return_value = mock_artifacts

        # Act
        result = handler.get_faces_by_cluster(video_id, cluster_id)

        # Assert
        assert len(result) == 1
        assert result[0].artifact_id == "artifact-1"
        payload = json.loads(result[0].payload_json)
        assert payload["cluster_id"] == cluster_id

    def test_determine_model_profile(self, handler):
        """Test model profile determination."""
        assert handler._determine_model_profile("yolov8n-face.pt") == "fast"
        assert handler._determine_model_profile("yolov8s-face.pt") == "fast"
        assert handler._determine_model_profile("yolov8m-face.pt") == "balanced"
        assert handler._determine_model_profile("yolov8l-face.pt") == "high_quality"
        assert handler._determine_model_profile("yolov8x-face.pt") == "high_quality"
