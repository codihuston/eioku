"""Domain models for artifact envelope architecture."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ArtifactEnvelope:
    """
    Domain model for ArtifactEnvelope - canonical artifact storage.

    Represents a time-aligned ML output or metadata with complete provenance.
    All fields are required to ensure complete artifact metadata.
    """

    artifact_id: str
    asset_id: str  # video_id
    artifact_type: str  # e.g., "transcript.segment", "scene", "object.detection"
    schema_version: int
    span_start_ms: int
    span_end_ms: int
    payload_json: str  # JSON-serialized payload
    producer: str  # e.g., "whisper", "pyscenedetect"
    producer_version: str  # e.g., "3.0.0"
    model_profile: str  # "fast" | "balanced" | "high_quality"
    config_hash: str  # Hash of configuration used
    input_hash: str  # Hash of input data
    run_id: str
    created_at: datetime

    def __post_init__(self):
        """Validate required fields are non-null."""
        required_fields = [
            "artifact_id",
            "asset_id",
            "artifact_type",
            "schema_version",
            "span_start_ms",
            "span_end_ms",
            "payload_json",
            "producer",
            "producer_version",
            "model_profile",
            "config_hash",
            "input_hash",
            "run_id",
            "created_at",
        ]

        for field in required_fields:
            value = getattr(self, field)
            if value is None:
                raise ValueError(f"Required field '{field}' cannot be None")

        # Validate time span
        if self.span_start_ms < 0:
            raise ValueError("span_start_ms must be non-negative")
        if self.span_end_ms < 0:
            raise ValueError("span_end_ms must be non-negative")
        if self.span_start_ms > self.span_end_ms:
            raise ValueError("span_start_ms must be <= span_end_ms")

        # Validate schema version
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")

    def get_duration_ms(self) -> int:
        """Get duration of artifact span in milliseconds."""
        return self.span_end_ms - self.span_start_ms

    def overlaps(self, start_ms: int, end_ms: int) -> bool:
        """Check if artifact span overlaps with given time range."""
        return self.span_start_ms < end_ms and self.span_end_ms > start_ms


@dataclass
class Run:
    """
    Domain model for Run - pipeline execution tracking.

    Tracks the lifecycle of a pipeline execution that produces artifacts.
    """

    run_id: str
    asset_id: str
    pipeline_profile: str
    started_at: datetime
    status: str  # "running" | "completed" | "failed"
    finished_at: datetime | None = None
    error: str | None = None

    def __post_init__(self):
        """Validate required fields and status."""
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if not self.asset_id:
            raise ValueError("asset_id cannot be empty")
        if not self.pipeline_profile:
            raise ValueError("pipeline_profile cannot be empty")
        if not self.started_at:
            raise ValueError("started_at cannot be None")

        valid_statuses = ["running", "completed", "failed"]
        if self.status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")

    def is_running(self) -> bool:
        """Check if run is currently running."""
        return self.status == "running"

    def is_completed(self) -> bool:
        """Check if run completed successfully."""
        return self.status == "completed"

    def is_failed(self) -> bool:
        """Check if run failed."""
        return self.status == "failed"

    def complete(self, finished_at: datetime) -> None:
        """Mark run as completed."""
        self.status = "completed"
        self.finished_at = finished_at

    def fail(self, error: str, finished_at: datetime) -> None:
        """Mark run as failed with error message."""
        self.status = "failed"
        self.error = error
        self.finished_at = finished_at


@dataclass
class SelectionPolicy:
    """
    Domain model for SelectionPolicy - artifact version selection rules.

    Defines which artifact version to use when multiple versions exist.
    """

    asset_id: str
    artifact_type: str
    mode: str  # "default" | "pinned" | "latest" | "profile" | "best_quality"
    preferred_profile: str | None = None
    pinned_run_id: str | None = None
    pinned_artifact_id: str | None = None
    updated_at: datetime | None = None

    def __post_init__(self):
        """Validate selection policy configuration."""
        if not self.asset_id:
            raise ValueError("asset_id cannot be empty")
        if not self.artifact_type:
            raise ValueError("artifact_type cannot be empty")

        valid_modes = ["default", "pinned", "latest", "profile", "best_quality"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")

        # Validate mode-specific requirements
        if self.mode == "profile" and not self.preferred_profile:
            raise ValueError("preferred_profile required when mode is 'profile'")
        if self.mode == "pinned" and not self.pinned_run_id:
            raise ValueError("pinned_run_id required when mode is 'pinned'")

    def is_default(self) -> bool:
        """Check if using default selection mode."""
        return self.mode == "default"

    def is_pinned(self) -> bool:
        """Check if using pinned selection mode."""
        return self.mode == "pinned"

    def is_profile_based(self) -> bool:
        """Check if using profile-based selection."""
        return self.mode == "profile"
