"""Tests for artifact domain models."""

from datetime import datetime

import pytest

from src.domain.artifacts import ArtifactEnvelope, Run, SelectionPolicy


class TestArtifactEnvelope:
    """Tests for ArtifactEnvelope domain model."""

    def test_create_valid_artifact_envelope(self):
        """Test creating a valid artifact envelope."""
        artifact = ArtifactEnvelope(
            artifact_id="art_123",
            asset_id="video_456",
            artifact_type="transcript.segment",
            schema_version=1,
            span_start_ms=1000,
            span_end_ms=2000,
            payload_json='{"text": "Hello world"}',
            producer="whisper",
            producer_version="3.0.0",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_789",
            created_at=datetime.utcnow(),
        )

        assert artifact.artifact_id == "art_123"
        assert artifact.asset_id == "video_456"
        assert artifact.artifact_type == "transcript.segment"
        assert artifact.get_duration_ms() == 1000

    def test_artifact_envelope_requires_all_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValueError, match="Required field"):
            ArtifactEnvelope(
                artifact_id="art_123",
                asset_id="video_456",
                artifact_type="transcript.segment",
                schema_version=1,
                span_start_ms=1000,
                span_end_ms=2000,
                payload_json='{"text": "Hello world"}',
                producer="whisper",
                producer_version="3.0.0",
                model_profile="balanced",
                config_hash="abc123",
                input_hash="def456",
                run_id=None,  # Missing required field
                created_at=datetime.utcnow(),
            )

    def test_artifact_envelope_validates_time_span(self):
        """Test that time span is validated."""
        with pytest.raises(ValueError, match="span_start_ms must be <= span_end_ms"):
            ArtifactEnvelope(
                artifact_id="art_123",
                asset_id="video_456",
                artifact_type="transcript.segment",
                schema_version=1,
                span_start_ms=2000,  # Start after end
                span_end_ms=1000,
                payload_json='{"text": "Hello world"}',
                producer="whisper",
                producer_version="3.0.0",
                model_profile="balanced",
                config_hash="abc123",
                input_hash="def456",
                run_id="run_789",
                created_at=datetime.utcnow(),
            )

    def test_artifact_envelope_overlaps(self):
        """Test overlap detection."""
        artifact = ArtifactEnvelope(
            artifact_id="art_123",
            asset_id="video_456",
            artifact_type="transcript.segment",
            schema_version=1,
            span_start_ms=1000,
            span_end_ms=2000,
            payload_json='{"text": "Hello world"}',
            producer="whisper",
            producer_version="3.0.0",
            model_profile="balanced",
            config_hash="abc123",
            input_hash="def456",
            run_id="run_789",
            created_at=datetime.utcnow(),
        )

        # Overlapping ranges
        assert artifact.overlaps(500, 1500) is True
        assert artifact.overlaps(1500, 2500) is True
        assert artifact.overlaps(1200, 1800) is True

        # Non-overlapping ranges
        assert artifact.overlaps(0, 1000) is False
        assert artifact.overlaps(2000, 3000) is False


class TestRun:
    """Tests for Run domain model."""

    def test_create_valid_run(self):
        """Test creating a valid run."""
        run = Run(
            run_id="run_123",
            asset_id="video_456",
            pipeline_profile="balanced",
            started_at=datetime.utcnow(),
            status="running",
        )

        assert run.run_id == "run_123"
        assert run.is_running() is True
        assert run.is_completed() is False

    def test_run_complete(self):
        """Test completing a run."""
        run = Run(
            run_id="run_123",
            asset_id="video_456",
            pipeline_profile="balanced",
            started_at=datetime.utcnow(),
            status="running",
        )

        finished_at = datetime.utcnow()
        run.complete(finished_at)

        assert run.is_completed() is True
        assert run.finished_at == finished_at

    def test_run_fail(self):
        """Test failing a run."""
        run = Run(
            run_id="run_123",
            asset_id="video_456",
            pipeline_profile="balanced",
            started_at=datetime.utcnow(),
            status="running",
        )

        finished_at = datetime.utcnow()
        run.fail("Test error", finished_at)

        assert run.is_failed() is True
        assert run.error == "Test error"
        assert run.finished_at == finished_at


class TestSelectionPolicy:
    """Tests for SelectionPolicy domain model."""

    def test_create_valid_selection_policy(self):
        """Test creating a valid selection policy."""
        policy = SelectionPolicy(
            asset_id="video_456",
            artifact_type="transcript.segment",
            mode="latest",
        )

        assert policy.asset_id == "video_456"
        assert policy.is_default() is False

    def test_profile_mode_requires_preferred_profile(self):
        """Test that profile mode requires preferred_profile."""
        with pytest.raises(ValueError, match="preferred_profile required"):
            SelectionPolicy(
                asset_id="video_456",
                artifact_type="transcript.segment",
                mode="profile",
                # Missing preferred_profile
            )

    def test_pinned_mode_requires_pinned_run_id(self):
        """Test that pinned mode requires pinned_run_id."""
        with pytest.raises(ValueError, match="pinned_run_id required"):
            SelectionPolicy(
                asset_id="video_456",
                artifact_type="transcript.segment",
                mode="pinned",
                # Missing pinned_run_id
            )
