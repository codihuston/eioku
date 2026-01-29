"""Schema for video metadata artifact type version 1."""

from pydantic import BaseModel, Field


class MetadataV1(BaseModel):
    """
    Payload schema for video metadata artifacts.

    Represents standardized metadata extracted from video files including
    GPS coordinates, camera information, file properties, temporal data,
    and image information. All fields are optional as not all videos
    contain all metadata types.
    """

    # GPS coordinates (optional)
    latitude: float | None = Field(default=None, description="GPS latitude coordinate")
    longitude: float | None = Field(
        default=None, description="GPS longitude coordinate"
    )
    altitude: float | None = Field(default=None, description="GPS altitude in meters")

    # Image properties (optional)
    image_size: str | None = Field(
        default=None, description="Image dimensions (e.g., '1920x1080')"
    )
    megapixels: float | None = Field(
        default=None, ge=0.0, description="Megapixels of the image"
    )
    rotation: int | None = Field(
        default=None, description="Image rotation in degrees (0, 90, 180, 270)"
    )

    # Audio/Video properties (optional)
    avg_bitrate: str | None = Field(
        default=None, description="Average bitrate (e.g., '5000k')"
    )
    duration_seconds: float | None = Field(
        default=None, ge=0.0, description="Video duration in seconds"
    )
    frame_rate: float | None = Field(
        default=None, ge=0.0, description="Video frame rate in fps"
    )
    codec: str | None = Field(
        default=None, description="Video codec (e.g., 'h264', 'hevc')"
    )

    # File properties (optional)
    file_size: int | None = Field(default=None, ge=0, description="File size in bytes")
    file_type: str | None = Field(default=None, description="File type (e.g., 'video')")
    mime_type: str | None = Field(
        default=None, description="MIME type (e.g., 'video/mp4')"
    )

    # Camera properties (optional)
    camera_make: str | None = Field(
        default=None, description="Camera manufacturer (e.g., 'Canon')"
    )
    camera_model: str | None = Field(
        default=None, description="Camera model (e.g., 'EOS R5')"
    )

    # Temporal properties (optional)
    create_date: str | None = Field(
        default=None, description="Creation date in ISO format"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "altitude": 10.5,
                    "image_size": "1920x1080",
                    "megapixels": 2.07,
                    "rotation": 0,
                    "avg_bitrate": "5000k",
                    "duration_seconds": 120.5,
                    "frame_rate": 29.97,
                    "codec": "h264",
                    "file_size": 75000000,
                    "file_type": "video",
                    "mime_type": "video/mp4",
                    "camera_make": "Canon",
                    "camera_model": "EOS R5",
                    "create_date": "2024-01-15T10:30:00Z",
                }
            ]
        }
    }
