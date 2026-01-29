"""Tests for projection sync service."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.domain.artifacts import ArtifactEnvelope
from src.services.projection_sync_service import (
    ProjectionSyncError,
    ProjectionSyncService,
)


class TestProjectionSyncService:
    """Test ProjectionSyncService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_bind = Mock()
        self.mock_session.bind = self.mock_bind

        self.service = ProjectionSyncService(self.mock_session)

        # Test artifact
        self.transcript_artifact = ArtifactEnvelope(
            artifact_id="artifact_123",
            asset_id="video_123",
            artifact_type="transcript.segment",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=5000,
            payload_json='{"text": "Hello world", "confidence": 0.9, "language": "en"}',
            producer="whisper",
            producer_version="large-v3",
            model_profile="high_quality",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

    def test_sync_transcript_artifact_postgresql(self):
        """Test syncing transcript artifact to PostgreSQL FTS."""
        # Mock PostgreSQL dialect
        self.mock_bind.dialect.name = "postgresql"

        # Sync artifact
        self.service.sync_artifact(self.transcript_artifact)

        # Verify SQL was executed (commit happens in batch_create, not here)
        assert self.mock_session.execute.called

        # Verify the SQL contains the expected data
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["artifact_id"] == "artifact_123"
        assert params["asset_id"] == "video_123"
        assert params["start_ms"] == 0
        assert params["end_ms"] == 5000
        assert params["text"] == "Hello world"

    def test_sync_transcript_artifact_sqlite(self):
        """Test syncing transcript artifact to SQLite FTS5."""
        # Mock SQLite dialect
        self.mock_bind.dialect.name = "sqlite"

        # Sync artifact
        self.service.sync_artifact(self.transcript_artifact)

        # Verify SQL was executed twice (metadata + FTS5)
        # Commit happens in batch_create, not here
        assert self.mock_session.execute.call_count == 2

    def test_sync_artifact_with_invalid_type(self):
        """Test syncing artifact with unsupported type (should not fail)."""
        # Create artifact with unsupported type
        artifact = ArtifactEnvelope(
            artifact_id="artifact_456",
            asset_id="video_123",
            artifact_type="unsupported.type",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=5000,
            payload_json='{"data": "test"}',
            producer="test",
            producer_version="1.0",
            model_profile="fast",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        # Should not raise error (just doesn't sync anything)
        self.service.sync_artifact(artifact)

        # Verify no SQL was executed
        assert not self.mock_session.execute.called

    def test_sync_artifact_database_error(self):
        """Test handling of database errors during sync."""
        # Mock database error
        self.mock_bind.dialect.name = "postgresql"
        self.mock_session.execute.side_effect = Exception("Database error")

        # Should raise ProjectionSyncError
        with pytest.raises(ProjectionSyncError) as exc_info:
            self.service.sync_artifact(self.transcript_artifact)

        assert "Failed to sync projection" in str(exc_info.value)
        assert "Database error" in str(exc_info.value)

    def test_sync_transcript_with_special_characters(self):
        """Test syncing transcript with special characters."""
        # Create artifact with special characters
        artifact = ArtifactEnvelope(
            artifact_id="artifact_789",
            asset_id="video_123",
            artifact_type="transcript.segment",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=5000,
            payload_json=(
                '{"text": "Hello \\"world\\" & <test>", '
                '"confidence": 0.9, "language": "en"}'
            ),
            producer="whisper",
            producer_version="large-v3",
            model_profile="high_quality",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        self.mock_bind.dialect.name = "postgresql"

        # Should not raise error
        self.service.sync_artifact(artifact)

        # Verify text was properly extracted
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["text"] == 'Hello "world" & <test>'

    def test_sync_scene_artifact(self):
        """Test syncing scene artifact to scene_ranges projection."""
        # Create scene artifact
        scene_artifact = ArtifactEnvelope(
            artifact_id="scene_123",
            asset_id="video_123",
            artifact_type="scene",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=5000,
            payload_json=(
                '{"scene_index": 1, "method": "content", '
                '"score": 0.95, "frame_number": 150}'
            ),
            producer="pyscenedetect",
            producer_version="0.6.1",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        # Sync artifact
        self.service.sync_artifact(scene_artifact)

        # Verify SQL was executed (commit happens in batch_create, not here)
        assert self.mock_session.execute.called

        # Verify the SQL contains the expected data
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["artifact_id"] == "scene_123"
        assert params["asset_id"] == "video_123"
        assert params["scene_index"] == 1
        assert params["start_ms"] == 0
        assert params["end_ms"] == 5000

    def test_sync_object_detection_artifact(self):
        """Test syncing object.detection artifact to object_labels projection."""
        # Create object detection artifact
        object_artifact = ArtifactEnvelope(
            artifact_id="object_123",
            asset_id="video_123",
            artifact_type="object.detection",
            schema_version=1,
            span_start_ms=1000,
            span_end_ms=1001,
            payload_json=(
                '{"label": "person", "confidence": 0.92, '
                '"bounding_box": {"x": 100, "y": 150, "width": 200, "height": 300}, '
                '"frame_number": 30}'
            ),
            producer="yolo",
            producer_version="yolov8n.pt",
            model_profile="fast",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        # Sync artifact
        self.service.sync_artifact(object_artifact)

        # Verify SQL was executed (commit happens in batch_create, not here)
        assert self.mock_session.execute.called

        # Verify the SQL contains the expected data
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["artifact_id"] == "object_123"
        assert params["asset_id"] == "video_123"
        assert params["label"] == "person"
        assert params["confidence"] == 0.92
        assert params["start_ms"] == 1000
        assert params["end_ms"] == 1001

    def test_sync_face_detection_artifact(self):
        """Test syncing face.detection artifact to face_clusters projection."""
        # Create face detection artifact
        face_artifact = ArtifactEnvelope(
            artifact_id="face_123",
            asset_id="video_123",
            artifact_type="face.detection",
            schema_version=1,
            span_start_ms=2000,
            span_end_ms=2001,
            payload_json=(
                '{"confidence": 0.95, '
                '"bounding_box": {"x": 250, "y": 100, "width": 150, "height": 180}, '
                '"cluster_id": "person_001", "frame_number": 60}'
            ),
            producer="yolo-face",
            producer_version="yolov8n-face.pt",
            model_profile="fast",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        # Sync artifact
        self.service.sync_artifact(face_artifact)

        # Verify SQL was executed (commit happens in batch_create, not here)
        assert self.mock_session.execute.called

        # Verify the SQL contains the expected data
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["artifact_id"] == "face_123"
        assert params["asset_id"] == "video_123"
        assert params["cluster_id"] == "person_001"
        assert params["confidence"] == 0.95
        assert params["start_ms"] == 2000
        assert params["end_ms"] == 2001

    def test_sync_ocr_text_artifact_postgresql(self):
        """Test syncing ocr.text artifact to PostgreSQL FTS."""
        # Create OCR text artifact
        ocr_artifact = ArtifactEnvelope(
            artifact_id="ocr_123",
            asset_id="video_123",
            artifact_type="ocr.text",
            schema_version=1,
            span_start_ms=3000,
            span_end_ms=3001,
            payload_json=(
                '{"text": "Welcome to the presentation", "confidence": 0.94, '
                '"bounding_box": [{"x": 100.0, "y": 50.0}, {"x": 400.0, "y": 50.0}, '
                '{"x": 400.0, "y": 100.0}, {"x": 100.0, "y": 100.0}], '
                '"language": "en", "frame_number": 90}'
            ),
            producer="easyocr",
            producer_version="easyocr_en",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        # Mock PostgreSQL dialect
        self.mock_bind.dialect.name = "postgresql"

        # Sync artifact
        self.service.sync_artifact(ocr_artifact)

        # Verify SQL was executed (commit happens in batch_create, not here)
        assert self.mock_session.execute.called

        # Verify the SQL contains the expected data
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["artifact_id"] == "ocr_123"
        assert params["asset_id"] == "video_123"
        assert params["start_ms"] == 3000
        assert params["end_ms"] == 3001
        assert params["text"] == "Welcome to the presentation"

    def test_sync_ocr_text_artifact_sqlite(self):
        """Test syncing ocr.text artifact to SQLite FTS5."""
        # Create OCR text artifact
        ocr_artifact = ArtifactEnvelope(
            artifact_id="ocr_456",
            asset_id="video_123",
            artifact_type="ocr.text",
            schema_version=1,
            span_start_ms=4000,
            span_end_ms=4001,
            payload_json=(
                '{"text": "Chapter 1", "confidence": 0.89, '
                '"bounding_box": [{"x": 50.0, "y": 25.0}, {"x": 200.0, "y": 25.0}, '
                '{"x": 200.0, "y": 75.0}, {"x": 50.0, "y": 75.0}], '
                '"language": "en", "frame_number": 120}'
            ),
            producer="easyocr",
            producer_version="easyocr_en",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        # Mock SQLite dialect
        self.mock_bind.dialect.name = "sqlite"

        # Sync artifact
        self.service.sync_artifact(ocr_artifact)

        # Verify SQL was executed twice (metadata + FTS5)
        # Commit happens in batch_create, not here
        assert self.mock_session.execute.call_count == 2

    def test_sync_video_metadata_with_gps_postgresql(self):
        """Test syncing video.metadata artifact with GPS to PostgreSQL."""
        # Create metadata artifact with GPS coordinates
        metadata_artifact = ArtifactEnvelope(
            artifact_id="metadata_001",
            asset_id="video_123",
            artifact_type="video.metadata",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=120000,
            payload_json=(
                '{"latitude": 40.7128, "longitude": -74.0060, "altitude": 10.5, '
                '"duration_seconds": 120.0, "file_size": 75000000, '
                '"mime_type": "video/mp4", "camera_make": "Canon", '
                '"camera_model": "EOS R5"}'
            ),
            producer="pyexiftool",
            producer_version="0.5.5",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_123",
            created_at=datetime.utcnow(),
        )

        # Mock PostgreSQL dialect
        self.mock_bind.dialect.name = "postgresql"

        # Sync artifact
        self.service.sync_artifact(metadata_artifact)

        # Verify SQL was executed
        assert self.mock_session.execute.called

        # Verify the SQL contains the expected GPS data
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["artifact_id"] == "metadata_001"
        assert params["video_id"] == "video_123"
        assert params["latitude"] == 40.7128
        assert params["longitude"] == -74.0060
        assert params["altitude"] == 10.5

    def test_sync_video_metadata_with_gps_sqlite(self):
        """Test syncing video.metadata artifact with GPS to SQLite."""
        # Create metadata artifact with GPS coordinates
        metadata_artifact = ArtifactEnvelope(
            artifact_id="metadata_002",
            asset_id="video_456",
            artifact_type="video.metadata",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=90000,
            payload_json=(
                '{"latitude": 51.5074, "longitude": -0.1278, "altitude": 5.0, '
                '"duration_seconds": 90.0, "file_size": 50000000}'
            ),
            producer="pyexiftool",
            producer_version="0.5.5",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_456",
            created_at=datetime.utcnow(),
        )

        # Mock SQLite dialect
        self.mock_bind.dialect.name = "sqlite"

        # Sync artifact
        self.service.sync_artifact(metadata_artifact)

        # Verify SQL was executed
        assert self.mock_session.execute.called

        # Verify the SQL contains the expected GPS data
        call_args = self.mock_session.execute.call_args
        params = call_args[0][1]
        assert params["latitude"] == 51.5074
        assert params["longitude"] == -0.1278
        assert params["altitude"] == 5.0

    def test_sync_video_metadata_without_gps(self):
        """Test syncing video.metadata artifact without GPS coordinates."""
        # Create metadata artifact without GPS
        metadata_artifact = ArtifactEnvelope(
            artifact_id="metadata_003",
            asset_id="video_789",
            artifact_type="video.metadata",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=60000,
            payload_json=(
                '{"duration_seconds": 60.0, "file_size": 40000000, '
                '"mime_type": "video/mp4"}'
            ),
            producer="pyexiftool",
            producer_version="0.5.5",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_789",
            created_at=datetime.utcnow(),
        )

        # Mock PostgreSQL dialect
        self.mock_bind.dialect.name = "postgresql"

        # Sync artifact
        self.service.sync_artifact(metadata_artifact)

        # Verify SQL was NOT executed (no GPS coordinates)
        assert not self.mock_session.execute.called

    def test_sync_video_metadata_invalid_latitude(self):
        """Test error handling for invalid latitude."""
        # Create metadata artifact with invalid latitude
        metadata_artifact = ArtifactEnvelope(
            artifact_id="metadata_004",
            asset_id="video_999",
            artifact_type="video.metadata",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=60000,
            payload_json=(
                '{"latitude": 95.0, "longitude": -74.0060, "altitude": 10.5}'
            ),
            producer="pyexiftool",
            producer_version="0.5.5",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_999",
            created_at=datetime.utcnow(),
        )

        # Mock PostgreSQL dialect
        self.mock_bind.dialect.name = "postgresql"

        # Should raise ProjectionSyncError
        with pytest.raises(ProjectionSyncError, match="Invalid latitude"):
            self.service.sync_artifact(metadata_artifact)

    def test_sync_video_metadata_invalid_longitude(self):
        """Test error handling for invalid longitude."""
        # Create metadata artifact with invalid longitude
        metadata_artifact = ArtifactEnvelope(
            artifact_id="metadata_005",
            asset_id="video_888",
            artifact_type="video.metadata",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=60000,
            payload_json=(
                '{"latitude": 40.7128, "longitude": 200.0, "altitude": 10.5}'
            ),
            producer="pyexiftool",
            producer_version="0.5.5",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_888",
            created_at=datetime.utcnow(),
        )

        # Mock PostgreSQL dialect
        self.mock_bind.dialect.name = "postgresql"

        # Should raise ProjectionSyncError
        with pytest.raises(ProjectionSyncError, match="Invalid longitude"):
            self.service.sync_artifact(metadata_artifact)

    def test_sync_video_metadata_partial_gps(self):
        """Test that partial GPS coordinates (only latitude) are skipped."""
        # Create metadata artifact with only latitude
        metadata_artifact = ArtifactEnvelope(
            artifact_id="metadata_006",
            asset_id="video_777",
            artifact_type="video.metadata",
            schema_version=1,
            span_start_ms=0,
            span_end_ms=60000,
            payload_json='{"latitude": 40.7128, "duration_seconds": 60.0}',
            producer="pyexiftool",
            producer_version="0.5.5",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_777",
            created_at=datetime.utcnow(),
        )

        # Mock PostgreSQL dialect
        self.mock_bind.dialect.name = "postgresql"

        # Sync artifact
        self.service.sync_artifact(metadata_artifact)

        # Verify SQL was NOT executed (incomplete GPS)
        assert not self.mock_session.execute.called
