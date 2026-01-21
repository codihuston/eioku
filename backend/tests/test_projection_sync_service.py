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

        # Verify SQL was executed
        assert self.mock_session.execute.called
        assert self.mock_session.commit.called

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
        assert self.mock_session.execute.call_count == 2
        assert self.mock_session.commit.called

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
