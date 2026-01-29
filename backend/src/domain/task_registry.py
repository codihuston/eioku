"""Task registry defining language behavior for each task type."""

from enum import Enum


class LanguageMode(Enum):
    """Language mode for a task type."""

    NONE = "none"  # Language not applicable (e.g., face_detection)
    REQUIRED = "required"  # Language is required, one task per language (e.g., ocr)
    OPTIONAL = (
        "optional"  # Language is optional, NULL = auto-detect (e.g., transcription)
    )


# Registry mapping task types to their language behavior
TASK_REGISTRY: dict[str, LanguageMode] = {
    "ocr": LanguageMode.REQUIRED,
    "transcription": LanguageMode.OPTIONAL,
    "face_detection": LanguageMode.NONE,
    "object_detection": LanguageMode.NONE,
    "place_detection": LanguageMode.NONE,
    "scene_detection": LanguageMode.NONE,
}


def is_language_required(task_type: str) -> bool:
    """Check if a task type requires a language to be specified."""
    return TASK_REGISTRY.get(task_type) == LanguageMode.REQUIRED


def is_language_optional(task_type: str) -> bool:
    """Check if a task type supports optional language specification."""
    return TASK_REGISTRY.get(task_type) == LanguageMode.OPTIONAL


def is_language_agnostic(task_type: str) -> bool:
    """Check if a task type does not use language at all."""
    return TASK_REGISTRY.get(task_type) == LanguageMode.NONE


def get_task_types() -> list[str]:
    """Get all registered task types."""
    return list(TASK_REGISTRY.keys())
