"""Tests for metadata artifact schema."""

import pytest
from pydantic import ValidationError

from src.domain.schema_initialization import register_all_schemas
from src.domain.schema_registry import SchemaRegistry
from src.domain.schemas import MetadataV1


class TestMetadataV1Schema:
    """Tests for MetadataV1 schema validation."""

    def setup_method(self):
        """Clear registry before each test."""
        SchemaRegistry.clear()

    def test_metadata_schema_with_all_fields(self):
        """Test MetadataV1 schema with all fields populated."""
        payload = {
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

        metadata = MetadataV1(**payload)

        assert metadata.latitude == 40.7128
        assert metadata.longitude == -74.0060
        assert metadata.altitude == 10.5
        assert metadata.image_size == "1920x1080"
        assert metadata.megapixels == 2.07
        assert metadata.rotation == 0
        assert metadata.avg_bitrate == "5000k"
        assert metadata.duration_seconds == 120.5
        assert metadata.frame_rate == 29.97
        assert metadata.codec == "h264"
        assert metadata.file_size == 75000000
        assert metadata.file_type == "video"
        assert metadata.mime_type == "video/mp4"
        assert metadata.camera_make == "Canon"
        assert metadata.camera_model == "EOS R5"
        assert metadata.create_date == "2024-01-15T10:30:00Z"

    def test_metadata_schema_with_only_gps_fields(self):
        """Test MetadataV1 schema with only GPS fields."""
        payload = {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "altitude": 10.5,
        }

        metadata = MetadataV1(**payload)

        assert metadata.latitude == 40.7128
        assert metadata.longitude == -74.0060
        assert metadata.altitude == 10.5
        assert metadata.image_size is None
        assert metadata.camera_make is None

    def test_metadata_schema_with_empty_payload(self):
        """Test MetadataV1 schema with empty payload (all optional)."""
        payload = {}

        metadata = MetadataV1(**payload)

        assert metadata.latitude is None
        assert metadata.longitude is None
        assert metadata.altitude is None
        assert metadata.image_size is None
        assert metadata.megapixels is None
        assert metadata.rotation is None
        assert metadata.avg_bitrate is None
        assert metadata.duration_seconds is None
        assert metadata.frame_rate is None
        assert metadata.codec is None
        assert metadata.file_size is None
        assert metadata.file_type is None
        assert metadata.mime_type is None
        assert metadata.camera_make is None
        assert metadata.camera_model is None
        assert metadata.create_date is None

    def test_metadata_schema_with_camera_fields(self):
        """Test MetadataV1 schema with camera fields."""
        payload = {
            "camera_make": "Canon",
            "camera_model": "EOS R5",
        }

        metadata = MetadataV1(**payload)

        assert metadata.camera_make == "Canon"
        assert metadata.camera_model == "EOS R5"

    def test_metadata_schema_with_file_fields(self):
        """Test MetadataV1 schema with file fields."""
        payload = {
            "file_size": 75000000,
            "file_type": "video",
            "mime_type": "video/mp4",
        }

        metadata = MetadataV1(**payload)

        assert metadata.file_size == 75000000
        assert metadata.file_type == "video"
        assert metadata.mime_type == "video/mp4"

    def test_metadata_schema_with_temporal_fields(self):
        """Test MetadataV1 schema with temporal fields."""
        payload = {
            "duration_seconds": 120.5,
            "frame_rate": 29.97,
            "create_date": "2024-01-15T10:30:00Z",
        }

        metadata = MetadataV1(**payload)

        assert metadata.duration_seconds == 120.5
        assert metadata.frame_rate == 29.97
        assert metadata.create_date == "2024-01-15T10:30:00Z"

    def test_metadata_schema_with_image_fields(self):
        """Test MetadataV1 schema with image fields."""
        payload = {
            "image_size": "1920x1080",
            "megapixels": 2.07,
            "rotation": 90,
        }

        metadata = MetadataV1(**payload)

        assert metadata.image_size == "1920x1080"
        assert metadata.megapixels == 2.07
        assert metadata.rotation == 90

    def test_metadata_schema_negative_megapixels_raises_error(self):
        """Test that negative megapixels raises validation error."""
        payload = {
            "megapixels": -1.0,
        }

        with pytest.raises(ValidationError):
            MetadataV1(**payload)

    def test_metadata_schema_negative_duration_raises_error(self):
        """Test that negative duration raises validation error."""
        payload = {
            "duration_seconds": -10.0,
        }

        with pytest.raises(ValidationError):
            MetadataV1(**payload)

    def test_metadata_schema_negative_frame_rate_raises_error(self):
        """Test that negative frame rate raises validation error."""
        payload = {
            "frame_rate": -29.97,
        }

        with pytest.raises(ValidationError):
            MetadataV1(**payload)

    def test_metadata_schema_negative_file_size_raises_error(self):
        """Test that negative file size raises validation error."""
        payload = {
            "file_size": -1000,
        }

        with pytest.raises(ValidationError):
            MetadataV1(**payload)

    def test_metadata_schema_serialization(self):
        """Test MetadataV1 schema serialization to JSON."""
        payload = {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "camera_make": "Canon",
        }

        metadata = MetadataV1(**payload)
        json_str = metadata.model_dump_json()

        assert '"latitude":40.7128' in json_str
        assert '"longitude":-74.006' in json_str
        assert '"camera_make":"Canon"' in json_str

    def test_metadata_schema_deserialization(self):
        """Test MetadataV1 schema deserialization from JSON."""
        json_str = (
            '{"latitude": 40.7128, "longitude": -74.0060, "camera_make": "Canon"}'
        )

        metadata = MetadataV1.model_validate_json(json_str)

        assert metadata.latitude == 40.7128
        assert metadata.longitude == -74.0060
        assert metadata.camera_make == "Canon"


class TestMetadataSchemaRegistration:
    """Tests for MetadataV1 schema registration."""

    def setup_method(self):
        """Clear registry before each test."""
        SchemaRegistry.clear()

    def test_metadata_schema_registration(self):
        """Test that MetadataV1 schema can be registered."""
        SchemaRegistry.register("video.metadata", 1, MetadataV1)

        assert SchemaRegistry.is_registered("video.metadata", 1) is True

    def test_metadata_schema_retrieval(self):
        """Test that registered MetadataV1 schema can be retrieved."""
        SchemaRegistry.register("video.metadata", 1, MetadataV1)

        schema = SchemaRegistry.get_schema("video.metadata", 1)
        assert schema == MetadataV1

    def test_metadata_schema_validation_via_registry(self):
        """Test validating metadata payload via registry."""
        SchemaRegistry.register("video.metadata", 1, MetadataV1)

        payload = {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "camera_make": "Canon",
            "camera_model": "EOS R5",
        }

        validated = SchemaRegistry.validate("video.metadata", 1, payload)

        assert validated.latitude == 40.7128
        assert validated.longitude == -74.0060
        assert validated.camera_make == "Canon"
        assert validated.camera_model == "EOS R5"

    def test_metadata_schema_serialization_via_registry(self):
        """Test serializing metadata payload via registry."""
        SchemaRegistry.register("video.metadata", 1, MetadataV1)

        payload = MetadataV1(
            latitude=40.7128,
            longitude=-74.0060,
            camera_make="Canon",
        )

        json_str = SchemaRegistry.serialize("video.metadata", 1, payload)

        assert '"latitude":40.7128' in json_str
        assert '"longitude":-74.006' in json_str
        assert '"camera_make":"Canon"' in json_str

    def test_metadata_schema_registered_in_initialization(self):
        """Test that MetadataV1 schema is registered during initialization."""
        register_all_schemas()

        assert SchemaRegistry.is_registered("video.metadata", 1) is True
        assert SchemaRegistry.get_schema("video.metadata", 1) == MetadataV1

    def test_metadata_schema_validation_after_initialization(self):
        """Test validating metadata after schema initialization."""
        register_all_schemas()

        payload = {
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

        validated = SchemaRegistry.validate("video.metadata", 1, payload)

        assert validated.latitude == 40.7128
        assert validated.longitude == -74.0060
        assert validated.altitude == 10.5
        assert validated.image_size == "1920x1080"
        assert validated.megapixels == 2.07
        assert validated.rotation == 0
        assert validated.avg_bitrate == "5000k"
        assert validated.duration_seconds == 120.5
        assert validated.frame_rate == 29.97
        assert validated.codec == "h264"
        assert validated.file_size == 75000000
        assert validated.file_type == "video"
        assert validated.mime_type == "video/mp4"
        assert validated.camera_make == "Canon"
        assert validated.camera_model == "EOS R5"
        assert validated.create_date == "2024-01-15T10:30:00Z"
