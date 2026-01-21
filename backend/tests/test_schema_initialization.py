"""Tests for schema initialization."""

from src.domain.schema_initialization import register_all_schemas
from src.domain.schema_registry import SchemaRegistry
from src.domain.schemas import (
    FaceDetectionV1,
    ObjectDetectionV1,
    OcrTextV1,
    PlaceClassificationV1,
    SceneV1,
    TranscriptSegmentV1,
)


class TestSchemaInitialization:
    """Tests for schema initialization."""

    def setup_method(self):
        """Clear registry before each test."""
        SchemaRegistry.clear()

    def test_register_all_schemas(self):
        """Test that all schemas are registered."""
        register_all_schemas()

        # Verify all schemas are registered
        assert SchemaRegistry.is_registered("transcript.segment", 1) is True
        assert SchemaRegistry.is_registered("scene", 1) is True
        assert SchemaRegistry.is_registered("object.detection", 1) is True
        assert SchemaRegistry.is_registered("face.detection", 1) is True
        assert SchemaRegistry.is_registered("place.classification", 1) is True
        assert SchemaRegistry.is_registered("ocr.text", 1) is True

    def test_registered_schemas_are_correct_types(self):
        """Test that registered schemas are the correct types."""
        register_all_schemas()

        assert SchemaRegistry.get_schema("transcript.segment", 1) == TranscriptSegmentV1
        assert SchemaRegistry.get_schema("scene", 1) == SceneV1
        assert SchemaRegistry.get_schema("object.detection", 1) == ObjectDetectionV1
        assert SchemaRegistry.get_schema("face.detection", 1) == FaceDetectionV1
        assert (
            SchemaRegistry.get_schema("place.classification", 1)
            == PlaceClassificationV1
        )
        assert SchemaRegistry.get_schema("ocr.text", 1) == OcrTextV1

    def test_schemas_can_validate_payloads(self):
        """Test that registered schemas can validate payloads."""
        register_all_schemas()

        # Test transcript.segment
        transcript_payload = {
            "text": "Hello world",
            "speaker": "Speaker 1",
            "confidence": 0.95,
            "language": "en",
        }
        validated = SchemaRegistry.validate("transcript.segment", 1, transcript_payload)
        assert validated.text == "Hello world"

        # Test scene
        scene_payload = {
            "scene_index": 0,
            "method": "content",
            "score": 0.85,
            "frame_number": 120,
        }
        validated = SchemaRegistry.validate("scene", 1, scene_payload)
        assert validated.scene_index == 0

        # Test object.detection
        object_payload = {
            "label": "person",
            "confidence": 0.92,
            "bounding_box": {"x": 100.0, "y": 150.0, "width": 200.0, "height": 300.0},
            "frame_number": 450,
        }
        validated = SchemaRegistry.validate("object.detection", 1, object_payload)
        assert validated.label == "person"
