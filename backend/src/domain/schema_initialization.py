"""Initialize and register all artifact schemas at application startup."""

from src.domain.schema_registry import SchemaRegistry
from src.domain.schemas import (
    FaceDetectionV1,
    ObjectDetectionV1,
    OcrTextV1,
    PlaceClassificationV1,
    SceneV1,
    TranscriptSegmentV1,
)


def register_all_schemas() -> None:
    """
    Register all artifact schemas with the schema registry.

    This function should be called during application startup to ensure
    all schemas are available for validation.

    This function is idempotent - it will only register schemas that are
    not already registered, making it safe to call multiple times.
    """
    # Register transcript.segment schemas
    if not SchemaRegistry.is_registered("transcript.segment", 1):
        SchemaRegistry.register("transcript.segment", 1, TranscriptSegmentV1)

    # Register scene schemas
    if not SchemaRegistry.is_registered("scene", 1):
        SchemaRegistry.register("scene", 1, SceneV1)

    # Register object.detection schemas
    if not SchemaRegistry.is_registered("object.detection", 1):
        SchemaRegistry.register("object.detection", 1, ObjectDetectionV1)

    # Register face.detection schemas
    if not SchemaRegistry.is_registered("face.detection", 1):
        SchemaRegistry.register("face.detection", 1, FaceDetectionV1)

    # Register place.classification schemas
    if not SchemaRegistry.is_registered("place.classification", 1):
        SchemaRegistry.register("place.classification", 1, PlaceClassificationV1)

    # Register ocr.text schemas
    if not SchemaRegistry.is_registered("ocr.text", 1):
        SchemaRegistry.register("ocr.text", 1, OcrTextV1)
