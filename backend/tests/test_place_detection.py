"""Tests for place detection service and task handler."""

import uuid
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.domain.artifacts import ArtifactEnvelope
from src.domain.models import Task, Video
from src.domain.schema_registry import SchemaRegistry
from src.domain.schemas.place_classification_v1 import (
    PlaceClassificationV1,
)
from src.services.place_detection_service import PlaceDetectionService
from src.services.place_detection_task_handler import PlaceDetectionTaskHandler

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_place_detection_service():
    """Create a mock place detection service."""
    service = Mock(spec=PlaceDetectionService)
    service.detect_places_in_video.return_value = [
        {
            "frame_number": 0,
            "timestamp": 0.0,
            "classifications": [
                {"label": "office", "confidence": 0.87},
                {"label": "conference_room", "confidence": 0.65},
                {"label": "classroom", "confidence": 0.42},
            ],
        },
        {
            "frame_number": 30,
            "timestamp": 1.0,
            "classifications": [
                {"label": "kitchen", "confidence": 0.92},
                {"label": "dining_room", "confidence": 0.55},
            ],
        },
    ]
    return service


@pytest.fixture
def mock_artifact_repository():
    """Create a mock artifact repository."""
    repo = Mock()
    repo.create.return_value = None
    repo.get_by_asset.return_value = []
    return repo


@pytest.fixture
def test_video():
    """Create a test video."""
    return Video(
        video_id=str(uuid.uuid4()),
        file_path="/path/to/test_video.mp4",
        filename="test_video.mp4",
        last_modified=datetime.utcnow(),
        status="pending",
        duration=10.0,
        file_size=1024000,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def test_task():
    """Create a test task."""
    return Task(
        task_id=str(uuid.uuid4()),
        video_id=str(uuid.uuid4()),
        task_type="place_detection",
        status="pending",
        created_at=datetime.utcnow(),
    )


# ============================================================================
# PlaceDetectionTaskHandler Tests
# ============================================================================


class TestPlaceDetectionTaskHandler:
    """Tests for PlaceDetectionTaskHandler."""

    def test_initialization(
        self, mock_artifact_repository, mock_place_detection_service
    ):
        """Test task handler initialization."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
            model_name="resnet18_places365",
            sample_rate=30,
            top_k=5,
        )

        assert handler.model_name == "resnet18_places365"
        assert handler.sample_rate == 30
        assert handler.top_k == 5
        assert handler.artifact_repository == mock_artifact_repository

    def test_process_place_detection_task_success(
        self,
        mock_artifact_repository,
        mock_place_detection_service,
        test_task,
        test_video,
    ):
        """Test successful place detection task processing."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        result = handler.process_place_detection_task(test_task, test_video)

        assert result is True
        # Should call batch_create once with 2 artifacts (one per frame result)
        mock_artifact_repository.batch_create.assert_called_once()
        call_args = mock_artifact_repository.batch_create.call_args
        artifacts = call_args[0][0]
        assert len(artifacts) == 2

        # Verify first artifact
        artifact = artifacts[0]
        assert isinstance(artifact, ArtifactEnvelope)
        assert artifact.asset_id == test_video.video_id
        assert artifact.artifact_type == "place.classification"
        assert artifact.schema_version == 1
        assert artifact.producer == "resnet_places365"
        assert artifact.model_profile == "fast"

    def test_process_place_detection_task_with_run_id(
        self,
        mock_artifact_repository,
        mock_place_detection_service,
        test_task,
        test_video,
    ):
        """Test place detection task processing with provided run_id."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        run_id = str(uuid.uuid4())
        result = handler.process_place_detection_task(test_task, test_video, run_id)

        assert result is True

        # Verify run_id is used
        call_args = mock_artifact_repository.batch_create.call_args
        artifacts = call_args[0][0]
        artifact = artifacts[0]
        assert artifact.run_id == run_id

    def test_process_place_detection_task_creates_valid_payloads(
        self,
        mock_artifact_repository,
        mock_place_detection_service,
        test_task,
        test_video,
    ):
        """Test that created artifacts have valid payloads."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        handler.process_place_detection_task(test_task, test_video)

        # Verify first artifact payload
        call_args = mock_artifact_repository.batch_create.call_args
        artifacts = call_args[0][0]
        artifact = artifacts[0]

        # Parse and validate payload
        payload = PlaceClassificationV1.model_validate_json(artifact.payload_json)
        assert payload.label == "office"
        assert payload.confidence == 0.87
        assert len(payload.alternative_labels) == 2
        assert payload.alternative_labels[0].label == "conference_room"
        assert payload.alternative_labels[0].confidence == 0.65
        assert payload.frame_number == 0

    def test_process_place_detection_task_failure(
        self, mock_artifact_repository, test_task, test_video
    ):
        """Test place detection task processing failure."""
        mock_service = Mock(spec=PlaceDetectionService)
        mock_service.detect_places_in_video.side_effect = Exception("Detection failed")

        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_service,
        )

        result = handler.process_place_detection_task(test_task, test_video)

        assert result is False
        mock_artifact_repository.create.assert_not_called()

    def test_get_detected_places(
        self, mock_artifact_repository, mock_place_detection_service
    ):
        """Test getting detected places for a video."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        video_id = str(uuid.uuid4())
        handler.get_detected_places(video_id)

        mock_artifact_repository.get_by_asset.assert_called_once_with(
            asset_id=video_id, artifact_type="place.classification"
        )

    def test_get_places_by_label(
        self, mock_artifact_repository, mock_place_detection_service
    ):
        """Test getting places filtered by label."""
        # Create mock artifacts
        payload1 = (
            '{"label": "office", "confidence": 0.87, '
            '"alternative_labels": [], "frame_number": 0}'
        )
        artifact1 = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id="video_123",
            artifact_type="place.classification",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=1,
            payload_json=payload1,
            producer="resnet_places365",
            producer_version="resnet18_places365",
            model_profile="fast",
            config_hash="abc123",
            input_hash="def456",
            run_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
        )

        payload2 = (
            '{"label": "kitchen", "confidence": 0.92, '
            '"alternative_labels": [], "frame_number": 30}'
        )
        artifact2 = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id="video_123",
            artifact_type="place.classification",
            schema_version=1,
            span_start_ms=1000,
            span_end_ms=1001,
            payload_json=payload2,
            producer="resnet_places365",
            producer_version="resnet18_places365",
            model_profile="fast",
            config_hash="abc123",
            input_hash="def456",
            run_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
        )

        mock_artifact_repository.get_by_asset.return_value = [artifact1, artifact2]

        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        # Get places with label "office"
        results = handler.get_places_by_label("video_123", "office")

        assert len(results) == 1
        assert results[0].artifact_id == artifact1.artifact_id

    def test_determine_model_profile(
        self, mock_artifact_repository, mock_place_detection_service
    ):
        """Test model profile determination."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        assert handler._determine_model_profile("resnet18_places365") == "fast"
        assert handler._determine_model_profile("resnet50_places365") == "balanced"
        assert handler._determine_model_profile("resnet152_places365") == "high_quality"

    def test_compute_config_hash(
        self, mock_artifact_repository, mock_place_detection_service
    ):
        """Test configuration hash computation."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        config = {"model_name": "resnet18_places365", "sample_rate": 30, "top_k": 5}
        hash1 = handler._compute_config_hash(config)
        hash2 = handler._compute_config_hash(config)

        # Same config should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 characters

    def test_compute_input_hash(
        self, mock_artifact_repository, mock_place_detection_service
    ):
        """Test input hash computation."""
        handler = PlaceDetectionTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            detection_service=mock_place_detection_service,
        )

        path = "/path/to/video.mp4"
        hash1 = handler._compute_input_hash(path)
        hash2 = handler._compute_input_hash(path)

        # Same path should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 characters
