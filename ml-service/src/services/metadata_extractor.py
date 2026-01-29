"""Metadata extractor for video files using pyexiftool.

This module implements metadata extraction from video files using pyexiftool,
extracting standardized Composite fields.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from video files using pyexiftool."""

    # Composite fields to extract (format-agnostic)
    COMPOSITE_FIELDS = [
        "GPSLatitude",
        "GPSLongitude",
        "GPSAltitude",
        "ImageSize",
        "Megapixels",
        "Rotation",
        "AvgBitrate",
        "Duration",
        "VideoFrameRate",
        "VideoCodec",
        "FileSize",
        "FileType",
        "MIMEType",
        "Make",
        "Model",
        "CreateDate",
    ]

    # Field name mappings from exiftool to output format
    FIELD_MAPPINGS = {
        "GPSLatitude": "latitude",
        "GPSLongitude": "longitude",
        "GPSAltitude": "altitude",
        "ImageSize": "image_size",
        "Megapixels": "megapixels",
        "Rotation": "rotation",
        "AvgBitrate": "avg_bitrate",
        "Duration": "duration_seconds",
        "VideoFrameRate": "frame_rate",
        "VideoCodec": "codec",
        "FileSize": "file_size",
        "FileType": "file_type",
        "MIMEType": "mime_type",
        "Make": "camera_make",
        "Model": "camera_model",
        "CreateDate": "create_date",
    }

    def __init__(self):
        """Initialize metadata extractor."""
        try:
            from exiftool import ExifToolHelper

            self.ExifToolHelper = ExifToolHelper
            logger.info("✓ Metadata extractor initialized with pyexiftool")
        except ImportError:
            logger.error("✗ pyexiftool not installed")
            raise

    def extract(self, video_path: str) -> dict:
        """Extract metadata from video file.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with extracted metadata fields (no null values)

        Raises:
            MetadataExtractionError: If extraction fails
        """
        try:
            logger.info(f"🎬 Extracting metadata from {video_path}")

            # Use exiftool to extract metadata
            with self.ExifToolHelper() as et:
                # Extract all metadata (no field filtering)
                metadata_list = et.get_metadata([video_path])
                metadata = metadata_list[0] if metadata_list else {}

            if not metadata:
                logger.warning(f"⚠️  No metadata found in {video_path}")
                return {}

            logger.info(
                f"✓ Extracted raw metadata from {video_path}: {len(metadata)} fields"
            )

            # Transform metadata to standardized format
            result = self._transform_metadata(metadata)

            logger.info(
                f"✓ Transformed metadata: {len(result)} fields extracted "
                f"from {video_path}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Error extracting metadata from {video_path}: {e}")
            raise MetadataExtractionError(
                f"Failed to extract metadata from {video_path}: {e}"
            )

    def _transform_metadata(self, metadata: dict) -> dict:
        """Transform raw exiftool metadata to standardized format.

        Args:
            metadata: Raw metadata dictionary from exiftool

        Returns:
            Dictionary with standardized metadata fields (no null values)
        """
        result = {}

        # Extract each composite field
        for exif_field, output_field in self.FIELD_MAPPINGS.items():
            value = self._extract_field(metadata, exif_field)

            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_value(output_field, value)
                if converted_value is not None:
                    result[output_field] = converted_value

        logger.debug(f"Transformed metadata: {result}")
        return result

    def _extract_field(self, metadata: dict, field_name: str) -> Any:
        """Extract a field from metadata.

        Args:
            metadata: Raw metadata dictionary
            field_name: Field name to extract

        Returns:
            Field value or None if not found
        """
        # Try exact match first
        if field_name in metadata:
            value = metadata[field_name]
            if value is not None and value != "":
                return value

        # Try with common prefixes if exact match fails
        for prefix in ["Composite:", "EXIF:", "QuickTime:", "File:"]:
            prefixed_field = f"{prefix}{field_name}"
            if prefixed_field in metadata:
                value = metadata[prefixed_field]
                if value is not None and value != "":
                    return value

        return None

    def _convert_value(self, output_field: str, value: Any) -> Any:
        """Convert exiftool value to appropriate Python type.

        Args:
            output_field: Output field name
            value: Raw value from exiftool

        Returns:
            Converted value or None if conversion fails
        """
        try:
            # Handle numeric fields
            if output_field in ["latitude", "longitude", "altitude", "megapixels"]:
                if isinstance(value, str):
                    # Parse string values like "40.7128" or "40 42 46.08"
                    if " " in value:
                        # DMS format: degrees minutes seconds
                        parts = value.split()
                        if len(parts) >= 3:
                            degrees = float(parts[0])
                            minutes = float(parts[1])
                            seconds = float(parts[2])
                            return degrees + minutes / 60 + seconds / 3600
                    else:
                        return float(value)
                return float(value)

            elif output_field in ["rotation", "file_size"]:
                return int(float(value)) if isinstance(value, str) else int(value)

            elif output_field == "frame_rate":
                if isinstance(value, str):
                    # Parse frame rate like "29.97" or "30"
                    return float(value.split()[0])
                return float(value)

            elif output_field == "duration_seconds":
                if isinstance(value, str):
                    # Parse duration like "120.5" or "2:00.5"
                    if ":" in value:
                        parts = value.split(":")
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                        return minutes * 60 + seconds
                    else:
                        return float(value)
                return float(value)

            # String fields - return as-is
            return str(value)

        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️  Failed to convert {output_field} value '{value}': {e}")
            return None


class MetadataExtractionError(Exception):
    """Exception raised when metadata extraction fails."""

    pass
