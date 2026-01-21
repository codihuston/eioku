"""Tests for OCR text detection service and task handler."""

import uuid
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.domain.artifacts import ArtifactEnvelope
from src.domain.models import Task, Video
from src.domain.schema_registry import SchemaRegistry
from src.domain.schemas.ocr_text_v1 import OcrTextV1
from src.services.ocr_service import OcrService
from src.services.ocr_task_handler import OcrTaskHandler

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_ocr_service():
    """Create a mock OCR service."""
    service = Mock(spec=OcrService)
    service.detect_text_in_video.return_value = [
        {
            "frame_number": 0,
            "timestamp": 0.0,
            "detections": [
                {
                    "text": "Welcome to the presentation",
                    "confidence": 0.94,
                    "bounding_box": [
                        {"x": 100.0, "y": 50.0},
                        {"x": 400.0, "y": 50.0},
                        {"x": 400.0, "y": 100.0},
                        {"x": 100.0, "y": 100.0},
                    ],
                    "language": "en",
                },
            ],
        },
        {
            "frame_number": 30,
            "timestamp": 1.0,
            "detections": [
                {
                    "text": "Chapter 1",
                    "confidence": 0.89,
                    "bounding_box": [
                        {"x": 50.0, "y": 25.0},
                        {"x": 200.0, "y": 25.0},
                        {"x": 200.0, "y": 75.0},
                        {"x": 50.0, "y": 75.0},
                    ],
                    "language": "en",
                },
                {
                    "text": "Introduction",
                    "confidence": 0.92,
                    "bounding_box": [
                        {"x": 50.0, "y": 100.0},
                        {"x": 250.0, "y": 100.0},
                        {"x": 250.0, "y": 150.0},
                        {"x": 50.0, "y": 150.0},
                    ],
                    "language": "en",
                },
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
        task_type="ocr_detection",
        status="pending",
        created_at=datetime.utcnow(),
    )


# ============================================================================
# OcrTaskHandler Tests
# ============================================================================


class TestOcrTaskHandler:
    """Tests for OcrTaskHandler."""

    def test_initialization(self, mock_artifact_repository, mock_ocr_service):
        """Test task handler initialization."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
            languages=["en"],
            sample_rate=30,
            gpu=False,
        )

        assert handler.languages == ["en"]
        assert handler.sample_rate == 30
        assert handler.gpu is False
        assert handler.artifact_repository == mock_artifact_repository

    def test_process_ocr_task_success(
        self,
        mock_artifact_repository,
        mock_ocr_service,
        test_task,
        test_video,
    ):
        """Test successful OCR task processing."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        result = handler.process_ocr_task(test_task, test_video)

        assert result is True
        # Should create 3 artifacts (1 from first frame, 2 from second frame)
        assert mock_artifact_repository.create.call_count == 3

        # Verify first artifact
        first_call = mock_artifact_repository.create.call_args_list[0]
        artifact = first_call[0][0]
        assert isinstance(artifact, ArtifactEnvelope)
        assert artifact.asset_id == test_video.video_id
        assert artifact.artifact_type == "ocr.text"
        assert artifact.schema_version == 1
        assert artifact.producer == "easyocr"
        assert artifact.model_profile == "balanced"

    def test_process_ocr_task_with_run_id(
        self,
        mock_artifact_repository,
        mock_ocr_service,
        test_task,
        test_video,
    ):
        """Test OCR task processing with provided run_id."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        run_id = str(uuid.uuid4())
        result = handler.process_ocr_task(test_task, test_video, run_id)

        assert result is True

        # Verify run_id is used
        first_call = mock_artifact_repository.create.call_args_list[0]
        artifact = first_call[0][0]
        assert artifact.run_id == run_id

    def test_process_ocr_task_creates_valid_payloads(
        self,
        mock_artifact_repository,
        mock_ocr_service,
        test_task,
        test_video,
    ):
        """Test that created artifacts have valid payloads."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        handler.process_ocr_task(test_task, test_video)

        # Verify first artifact payload
        first_call = mock_artifact_repository.create.call_args_list[0]
        artifact = first_call[0][0]

        # Parse and validate payload
        payload = OcrTextV1.model_validate_json(artifact.payload_json)
        assert payload.text == "Welcome to the presentation"
        assert payload.confidence == 0.94
        assert len(payload.bounding_box) == 4
        assert payload.bounding_box[0].x == 100.0
        assert payload.bounding_box[0].y == 50.0
        assert payload.language == "en"
        assert payload.frame_number == 0

    def test_process_ocr_task_failure(
        self, mock_artifact_repository, test_task, test_video
    ):
        """Test OCR task processing failure."""
        mock_service = Mock(spec=OcrService)
        mock_service.detect_text_in_video.side_effect = Exception("Detection failed")

        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_service,
        )

        result = handler.process_ocr_task(test_task, test_video)

        assert result is False
        mock_artifact_repository.create.assert_not_called()

    def test_get_detected_text(self, mock_artifact_repository, mock_ocr_service):
        """Test getting detected text for a video."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        video_id = str(uuid.uuid4())
        handler.get_detected_text(video_id)

        mock_artifact_repository.get_by_asset.assert_called_once_with(
            asset_id=video_id, artifact_type="ocr.text"
        )

    def test_get_text_by_content(self, mock_artifact_repository, mock_ocr_service):
        """Test getting text filtered by content."""
        # Create mock artifacts
        payload1 = (
            '{"text": "Welcome to the presentation", "confidence": 0.94, '
            '"bounding_box": [{"x": 100.0, "y": 50.0}, {"x": 400.0, "y": 50.0}, '
            '{"x": 400.0, "y": 100.0}, {"x": 100.0, "y": 100.0}], '
            '"language": "en", "frame_number": 0}'
        )
        artifact1 = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id="video_123",
            artifact_type="ocr.text",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=1,
            payload_json=payload1,
            producer="easyocr",
            producer_version="easyocr_en",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
        )

        payload2 = (
            '{"text": "Chapter 1", "confidence": 0.89, '
            '"bounding_box": [{"x": 50.0, "y": 25.0}, {"x": 200.0, "y": 25.0}, '
            '{"x": 200.0, "y": 75.0}, {"x": 50.0, "y": 75.0}], '
            '"language": "en", "frame_number": 30}'
        )
        artifact2 = ArtifactEnvelope(
            artifact_id=str(uuid.uuid4()),
            asset_id="video_123",
            artifact_type="ocr.text",
            schema_version=1,
            span_start_ms=1000,
            span_end_ms=1001,
            payload_json=payload2,
            producer="easyocr",
            producer_version="easyocr_en",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
        )

        mock_artifact_repository.get_by_asset.return_value = [artifact1, artifact2]

        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        # Get text containing "presentation"
        results = handler.get_text_by_content("video_123", "presentation")

        assert len(results) == 1
        assert results[0].artifact_id == artifact1.artifact_id

    def test_determine_model_profile(self, mock_artifact_repository, mock_ocr_service):
        """Test model profile determination."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        assert handler._determine_model_profile(gpu=True) == "fast"
        assert handler._determine_model_profile(gpu=False) == "balanced"

    def test_compute_config_hash(self, mock_artifact_repository, mock_ocr_service):
        """Test configuration hash computation."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        config = {"languages": ["en"], "sample_rate": 30, "gpu": False}
        hash1 = handler._compute_config_hash(config)
        hash2 = handler._compute_config_hash(config)

        # Same config should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 characters

    def test_compute_input_hash(self, mock_artifact_repository, mock_ocr_service):
        """Test input hash computation."""
        handler = OcrTaskHandler(
            artifact_repository=mock_artifact_repository,
            schema_registry=SchemaRegistry,
            ocr_service=mock_ocr_service,
        )

        path = "/path/to/video.mp4"
        hash1 = handler._compute_input_hash(path)
        hash2 = handler._compute_input_hash(path)

        # Same path should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 characters
