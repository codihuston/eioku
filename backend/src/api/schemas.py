from datetime import datetime

from pydantic import BaseModel, Field


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
