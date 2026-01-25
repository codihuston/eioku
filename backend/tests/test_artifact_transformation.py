"""Tests for artifact envelope transformation logic."""

import json
from uuid import uuid4

import pytest

from src.domain.artifact_transformation import (
    transform_to_envelopes,
    validate_ml_response,
)
from src.domain.artifacts import ArtifactEnvelope


class TestTransformToEnvelopes:
    """Test artifact envelope transformation."""

    def test_transform_object_detection_response(self):
        """Test transforming object detection ML response."""
        video_id = "video_123"
        task_id = "task_456"
        task_type = "object_detection"
        run_id = str(uuid4())
        config_hash = "config_abc123"
        input_hash = "input_def456"

        ml_response = {
            "detections": [
                {
                    "start_ms": 0,
                    "end_ms": 1000,
                    "label": "person",
                    "confidence": 0.95,
                    "bbox": [10, 20, 100, 200],
                },
                {
                    "start_ms": 1000,
                    "end_ms": 2000,
                    "label": "car",
                    "confidence": 0.87,
                    "bbox": [50, 60, 150, 160],
                },
            ]
        }

        envelopes = transform_to_envelopes(
            ml_response=ml_response,
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
            config_hash=config_hash,
            input_hash=input_hash,
            run_id=run_id,
        )

        assert len(envelopes) == 2
        assert all(isinstance(e, ArtifactEnvelope) for e in envelopes)

        # Verify first envelope
        assert envelopes[0].asset_id == video_id
        assert envelopes[0].artifact_type == "object_detection"
        assert envelopes[0].span_start_ms == 0
        assert envelopes[0].span_end_ms == 1000
        assert envelopes[0].config_hash == config_hash
        assert envelopes[0].input_hash == input_hash
        assert envelopes[0].run_id == run_id

        # Verify payload is JSON
        payload = json.loads(envelopes[0].payload_json)
        assert payload["label"] == "person"
        assert payload["confidence"] == 0.95

    def test_transform_transcription_response(self):
        """Test transforming transcription ML response."""
        video_id = "video_789"
        task_id = "task_012"
        task_type = "transcription"
        run_id = str(uuid4())

        ml_response = {
            "detections": [
                {
                    "start_ms": 0,
                    "end_ms": 5000,
                    "text": "Hello world",
                    "confidence": 0.92,
                },
                {
                    "start_ms": 5000,
                    "end_ms": 10000,
                    "text": "This is a test",
                    "confidence": 0.88,
                },
            ]
        }

        envelopes = transform_to_envelopes(
            ml_response=ml_response,
            task_id=task_id,
            video_id=video_id,
            task_type=task_type,
            config_hash="config_xyz",
            input_hash="input_xyz",
            run_id=run_id,
        )

        assert len(envelopes) == 2
        assert envelopes[0].artifact_type == "transcript_segment"
        assert envelopes[1].artifact_type == "transcript_segment"

    def test_transform_with_custom_producer_info(self):
        """Test transformation with custom producer metadata."""
        ml_response = {"detections": [{"start_ms": 0, "end_ms": 1000, "data": "test"}]}

        envelopes = transform_to_envelopes(
            ml_response=ml_response,
            task_id="task_123",
            video_id="video_123",
            task_type="object_detection",
            config_hash="config",
            input_hash="input",
            run_id="run_123",
            producer="custom-producer",
            producer_version="2.5.0",
            model_profile="high_quality",
        )

        assert len(envelopes) == 1
        assert envelopes[0].producer == "custom-producer"
        assert envelopes[0].producer_version == "2.5.0"
        assert envelopes[0].model_profile == "high_quality"

    def test_transform_empty_detections(self):
        """Test transformation with empty detections list."""
        ml_response = {"detections": []}

        envelopes = transform_to_envelopes(
            ml_response=ml_response,
            task_id="task_123",
            video_id="video_123",
            task_type="object_detection",
            config_hash="config",
            input_hash="input",
            run_id="run_123",
        )

        assert len(envelopes) == 0

    def test_transform_invalid_task_type(self):
        """Test error with invalid task type."""
        ml_response = {"detections": [{"start_ms": 0, "end_ms": 1000}]}

        with pytest.raises(ValueError, match="Unknown task type"):
            transform_to_envelopes(
                ml_response=ml_response,
                task_id="task_123",
                video_id="video_123",
                task_type="invalid_type",
                config_hash="config",
                input_hash="input",
                run_id="run_123",
            )

    def test_transform_empty_response(self):
        """Test error with empty response."""
        with pytest.raises(ValueError, match="ml_response cannot be empty"):
            transform_to_envelopes(
                ml_response={},
                task_id="task_123",
                video_id="video_123",
                task_type="object_detection",
                config_hash="config",
                input_hash="input",
                run_id="run_123",
            )

    def test_transform_invalid_time_span(self):
        """Test handling of invalid time spans."""
        ml_response = {
            "detections": [
                {
                    "start_ms": 1000,
                    "end_ms": 500,  # end < start
                    "data": "invalid",
                },
                {
                    "start_ms": 0,
                    "end_ms": 1000,
                    "data": "valid",
                },
            ]
        }

        envelopes = transform_to_envelopes(
            ml_response=ml_response,
            task_id="task_123",
            video_id="video_123",
            task_type="object_detection",
            config_hash="config",
            input_hash="input",
            run_id="run_123",
        )

        # Only valid detection should be transformed
        assert len(envelopes) == 1
        assert envelopes[0].span_start_ms == 0
        assert envelopes[0].span_end_ms == 1000

    def test_transform_negative_time_span(self):
        """Test handling of negative time spans."""
        ml_response = {
            "detections": [
                {
                    "start_ms": -100,
                    "end_ms": 1000,
                    "data": "negative start",
                },
                {
                    "start_ms": 0,
                    "end_ms": -100,
                    "data": "negative end",
                },
                {
                    "start_ms": 0,
                    "end_ms": 1000,
                    "data": "valid",
                },
            ]
        }

        envelopes = transform_to_envelopes(
            ml_response=ml_response,
            task_id="task_123",
            video_id="video_123",
            task_type="object_detection",
            config_hash="config",
            input_hash="input",
            run_id="run_123",
        )

        # Only valid detection should be transformed
        assert len(envelopes) == 1

    def test_transform_all_task_types(self):
        """Test transformation for all supported task types."""
        task_types = [
            "object_detection",
            "face_detection",
            "transcription",
            "ocr",
            "place_detection",
            "scene_detection",
        ]

        expected_artifact_types = {
            "object_detection": "object_detection",
            "face_detection": "face_detection",
            "transcription": "transcript_segment",
            "ocr": "ocr_detection",
            "place_detection": "place_classification",
            "scene_detection": "scene",
        }

        ml_response = {"detections": [{"start_ms": 0, "end_ms": 1000, "data": "test"}]}

        for task_type in task_types:
            envelopes = transform_to_envelopes(
                ml_response=ml_response,
                task_id="task_123",
                video_id="video_123",
                task_type=task_type,
                config_hash="config",
                input_hash="input",
                run_id="run_123",
            )

            assert len(envelopes) == 1
            assert envelopes[0].artifact_type == expected_artifact_types[task_type]

    def test_transform_preserves_detection_data(self):
        """Test that detection data is preserved in payload."""
        detection_data = {
            "start_ms": 0,
            "end_ms": 1000,
            "label": "person",
            "confidence": 0.95,
            "bbox": [10, 20, 100, 200],
            "custom_field": "custom_value",
        }

        ml_response = {"detections": [detection_data]}

        envelopes = transform_to_envelopes(
            ml_response=ml_response,
            task_id="task_123",
            video_id="video_123",
            task_type="object_detection",
            config_hash="config",
            input_hash="input",
            run_id="run_123",
        )

        assert len(envelopes) == 1
        payload = json.loads(envelopes[0].payload_json)

        # Verify all detection data is preserved
        for key, value in detection_data.items():
            assert payload[key] == value


class TestValidateMLResponse:
    """Test ML response validation."""

    def test_validate_valid_response(self):
        """Test validation of valid ML response."""
        ml_response = {
            "detections": [
                {"start_ms": 0, "end_ms": 1000, "label": "person"},
                {"start_ms": 1000, "end_ms": 2000, "label": "car"},
            ]
        }

        assert validate_ml_response(ml_response, "object_detection") is True

    def test_validate_empty_detections(self):
        """Test validation with empty detections."""
        ml_response = {"detections": []}

        assert validate_ml_response(ml_response, "object_detection") is True

    def test_validate_missing_detections_field(self):
        """Test validation with missing detections field."""
        ml_response = {"data": []}

        assert validate_ml_response(ml_response, "object_detection") is False

    def test_validate_detections_not_list(self):
        """Test validation when detections is not a list."""
        ml_response = {"detections": {"data": "invalid"}}

        assert validate_ml_response(ml_response, "object_detection") is False

    def test_validate_detection_not_dict(self):
        """Test validation when detection is not a dict."""
        ml_response = {"detections": ["invalid"]}

        assert validate_ml_response(ml_response, "object_detection") is False

    def test_validate_detection_missing_time_fields(self):
        """Test validation when detection missing time fields."""
        ml_response = {
            "detections": [{"label": "person"}]  # Missing start_ms, end_ms
        }

        assert validate_ml_response(ml_response, "object_detection") is False

    def test_validate_not_dict(self):
        """Test validation when response is not a dict."""
        assert validate_ml_response([], "object_detection") is False
        assert validate_ml_response("invalid", "object_detection") is False
        assert validate_ml_response(None, "object_detection") is False
