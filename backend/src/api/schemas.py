from datetime import datetime

from pydantic import BaseModel, Field


class JumpToSchema(BaseModel):
    """Schema for jump target timestamp."""

    start_ms: int = Field(..., description="Start timestamp in milliseconds")
    end_ms: int = Field(..., description="End timestamp in milliseconds")


class JumpResponseSchema(BaseModel):
    """Schema for jump navigation response."""

    jump_to: JumpToSchema = Field(..., description="Target timestamp to jump to")
    artifact_ids: list[str] = Field(..., description="List of artifact IDs at target")


class FindMatchSchema(BaseModel):
    """Schema for find match result."""

    jump_to: JumpToSchema = Field(..., description="Target timestamp to jump to")
    artifact_id: str = Field(..., description="Artifact ID containing the match")
    snippet: str = Field(..., description="Highlighted text snippet")
    source: str = Field(..., description="Source of match: transcript or ocr")


class FindResponseSchema(BaseModel):
    """Schema for find within video response."""

    matches: list[FindMatchSchema] = Field(..., description="List of matches found")


class ArtifactPayloadSchema(BaseModel):
    """Schema for artifact payload (flexible dict)."""

    class Config:
        extra = "allow"


class ArtifactResponseSchema(BaseModel):
    """Schema for artifact response."""

    artifact_id: str = Field(..., description="Unique artifact identifier")
    asset_id: str = Field(..., description="Asset (video) ID")
    artifact_type: str = Field(..., description="Type of artifact")
    schema_version: int = Field(..., description="Schema version")
    span_start_ms: int = Field(..., description="Start timestamp in milliseconds")
    span_end_ms: int = Field(..., description="End timestamp in milliseconds")
    payload: dict = Field(..., description="Artifact payload data")
    producer: str = Field(..., description="Producer name")
    producer_version: str = Field(..., description="Producer version")
    model_profile: str = Field(..., description="Model profile")
    run_id: str = Field(..., description="Run ID")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True


class VideoCreateSchema(BaseModel):
    """Schema for creating a new video."""

    video_id: str = Field(..., description="Unique video identifier")
    file_path: str = Field(..., description="Path to video file")
    filename: str = Field(..., description="Video filename")
    last_modified: datetime = Field(..., description="File last modified timestamp")
    duration: float | None = Field(None, description="Video duration in seconds")
    file_size: int | None = Field(None, description="File size in bytes")
    file_hash: str | None = Field(None, description="SHA-256 hash of video file")


class VideoResponseSchema(BaseModel):
    """Schema for video API responses."""

    video_id: str
    file_path: str
    filename: str
    last_modified: datetime
    status: str
    duration: float | None = None
    file_size: int | None = None
    file_hash: str | None = None
    processed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class VideoUpdateSchema(BaseModel):
    """Schema for updating video."""

    status: str | None = Field(None, description="Processing status")
    duration: float | None = Field(None, description="Video duration in seconds")
    file_size: int | None = Field(None, description="File size in bytes")
    processed_at: datetime | None = Field(
        None, description="Processing completion time"
    )


class ProfileInfoSchema(BaseModel):
    """Schema for model profile information."""

    profile: str = Field(..., description="Model profile name")
    artifact_count: int = Field(
        ..., description="Number of artifacts with this profile"
    )
    run_ids: list[str] = Field(..., description="List of run IDs for this profile")


class ProfilesResponseSchema(BaseModel):
    """Schema for profiles endpoint response."""

    video_id: str = Field(..., description="Video ID")
    artifact_type: str = Field(..., description="Artifact type")
    profiles: list[ProfileInfoSchema] = Field(..., description="Available profiles")


class RunInfoSchema(BaseModel):
    """Schema for run information."""

    run_id: str = Field(..., description="Run ID")
    created_at: datetime = Field(..., description="Creation timestamp of the run")
    artifact_count: int = Field(
        ..., description="Number of artifacts produced in this run"
    )
    model_profile: str | None = Field(
        None, description="Model profile used for this run"
    )
    language: str | None = Field(
        None, description="Language of artifacts in this run (if applicable)"
    )


class RunsResponseSchema(BaseModel):
    """Schema for runs endpoint response."""

    video_id: str = Field(..., description="Video ID")
    artifact_type: str = Field(..., description="Artifact type")
    runs: list[RunInfoSchema] = Field(..., description="Available runs")
