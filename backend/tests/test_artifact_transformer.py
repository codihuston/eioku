"""Unit tests for artifact transformation and schema validation.

Tests the ArtifactTransformer class which converts ML Service results
into validated ArtifactEnvelopes.
"""

import json

import pytest

from src.workers.artifact_transformer import ArtifactTransformer


class TestObjectDetectionTransformation:
    """Tests for object detection artifact transformation."""

    def test_valid_object_detection_transformation(self):
        """Test transforming valid object detection results."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "bounding_box": {
                        "x": 100.0,
                        "y": 150.0,
                        "width": 200.0,
                        "height": 300.0,
                    },
                    "frame_number": 450,
                },
                {
                    "label": "car",
                    "confidence": 0.87,
                    "bounding_box": {
                        "x": 400.0,
                        "y": 200.0,
                        "width": 150.0,
                        "height": 100.0,
                    },
                    "frame_number": 450,
                },
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_001",
            task_type="object_detection",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 2
        assert envelopes[0]["artifact_type"] == "object.detection"
        assert envelopes[0]["asset_id"] == "video_001"
        assert envelopes[0]["config_hash"] == "config_abc123"
        assert envelopes[0]["producer"] == "yolo"
        assert envelopes[0]["payload_json"]["label"] == "person"
        assert envelopes[0]["payload_json"]["confidence"] == 0.95

    def test_object_detection_missing_provenance(self):
        """Test that missing provenance fields raise ValueError."""
        ml_result = {
            "config_hash": "config_abc123",
            # Missing input_hash
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "bounding_box": {
                        "x": 100.0,
                        "y": 150.0,
                        "width": 200.0,
                        "height": 300.0,
                    },
                    "frame_number": 450,
                }
            ],
        }

        with pytest.raises(ValueError, match="missing required provenance fields"):
            ArtifactTransformer.transform_ml_result(
                task_id="task_001",
                task_type="object_detection",
                video_id="video_001",
                ml_result=ml_result,
            )

    def test_object_detection_invalid_schema(self):
        """Test that invalid detection data raises ValidationError."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "label": "person",
                    "confidence": 1.5,  # Invalid: > 1.0
                    "bounding_box": {
                        "x": 100.0,
                        "y": 150.0,
                        "width": 200.0,
                        "height": 300.0,
                    },
                    "frame_number": 450,
                }
            ],
        }

        with pytest.raises(ValueError, match="Invalid artifact payload"):
            ArtifactTransformer.transform_ml_result(
                task_id="task_001",
                task_type="object_detection",
                video_id="video_001",
                ml_result=ml_result,
            )


class TestFaceDetectionTransformation:
    """Tests for face detection artifact transformation."""

    def test_valid_face_detection_with_cluster(self):
        """Test transforming face detection with cluster ID."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "confidence": 0.98,
                    "bounding_box": {
                        "x": 150.0,
                        "y": 100.0,
                        "width": 120.0,
                        "height": 150.0,
                    },
                    "frame_number": 300,
                    "cluster_id": "face_cluster_001",
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_002",
            task_type="face_detection",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 1
        assert envelopes[0]["artifact_type"] == "face.detection"
        assert envelopes[0]["payload_json"]["cluster_id"] == "face_cluster_001"

    def test_face_detection_without_cluster(self):
        """Test face detection without cluster ID (optional field)."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "confidence": 0.92,
                    "bounding_box": {
                        "x": 150.0,
                        "y": 100.0,
                        "width": 120.0,
                        "height": 150.0,
                    },
                    "frame_number": 300,
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_002",
            task_type="face_detection",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 1
        assert envelopes[0]["payload_json"]["cluster_id"] is None


class TestTranscriptionTransformation:
    """Tests for transcription artifact transformation."""

    def test_valid_transcription_with_words(self):
        """Test transforming transcription with word-level details."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "whisper",
            "producer_version": "3.0",
            "model_profile": "balanced",
            "segments": [
                {
                    "text": "Hello, how are you?",
                    "start_ms": 1000,
                    "end_ms": 3500,
                    "confidence": 0.98,
                    "words": [
                        {
                            "word": "Hello",
                            "start": 1.0,
                            "end": 1.3,
                            "confidence": 0.99,
                        },
                        {
                            "word": "how",
                            "start": 1.4,
                            "end": 1.6,
                            "confidence": 0.98,
                        },
                    ],
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_003",
            task_type="transcription",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 1
        assert envelopes[0]["artifact_type"] == "transcription"
        assert envelopes[0]["span_start_ms"] == 1000
        assert envelopes[0]["span_end_ms"] == 3500
        assert envelopes[0]["payload_json"]["text"] == "Hello, how are you?"
        assert len(envelopes[0]["payload_json"]["words"]) == 2

    def test_transcription_without_words(self):
        """Test transcription without word-level details (optional)."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "whisper",
            "producer_version": "3.0",
            "model_profile": "balanced",
            "segments": [
                {
                    "text": "Hello world",
                    "start_ms": 500,
                    "end_ms": 2000,
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_003",
            task_type="transcription",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 1
        assert envelopes[0]["payload_json"]["words"] is None


class TestOCRTransformation:
    """Tests for OCR artifact transformation."""

    def test_valid_ocr_detection(self):
        """Test transforming valid OCR detection."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "easyocr",
            "producer_version": "1.7.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "text": "STOP",
                    "confidence": 0.92,
                    "polygon": [
                        {"x": 100.0, "y": 50.0},
                        {"x": 200.0, "y": 50.0},
                        {"x": 200.0, "y": 100.0},
                        {"x": 100.0, "y": 100.0},
                    ],
                    "frame_number": 450,
                    "language": "en",
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_004",
            task_type="ocr",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 1
        assert envelopes[0]["artifact_type"] == "ocr"
        assert envelopes[0]["payload_json"]["text"] == "STOP"
        assert len(envelopes[0]["payload_json"]["polygon"]) == 4

    def test_ocr_invalid_polygon(self):
        """Test that OCR with invalid polygon (< 3 points) fails."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "easyocr",
            "producer_version": "1.7.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "text": "STOP",
                    "confidence": 0.92,
                    "polygon": [
                        {"x": 100.0, "y": 50.0},
                        {"x": 200.0, "y": 50.0},
                    ],  # Only 2 points
                    "frame_number": 450,
                    "language": "en",
                }
            ],
        }

        with pytest.raises(ValueError, match="Invalid artifact payload"):
            ArtifactTransformer.transform_ml_result(
                task_id="task_004",
                task_type="ocr",
                video_id="video_001",
                ml_result=ml_result,
            )


class TestPlaceClassificationTransformation:
    """Tests for place classification artifact transformation."""

    def test_valid_place_classification(self):
        """Test transforming valid place classification."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "places365",
            "producer_version": "1.0.0",
            "model_profile": "balanced",
            "classifications": [
                {
                    "predictions": [
                        {"label": "beach", "confidence": 0.85},
                        {"label": "coast", "confidence": 0.12},
                        {"label": "ocean", "confidence": 0.02},
                    ],
                    "frame_number": 600,
                    "top_k": 3,
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_005",
            task_type="place_detection",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 1
        assert envelopes[0]["artifact_type"] == "place.classification"
        assert len(envelopes[0]["payload_json"]["predictions"]) == 3
        assert envelopes[0]["payload_json"]["predictions"][0]["label"] == "beach"


class TestSceneDetectionTransformation:
    """Tests for scene detection artifact transformation."""

    def test_valid_scene_detection(self):
        """Test transforming valid scene detection."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "scenedetect",
            "producer_version": "0.6.0",
            "model_profile": "balanced",
            "scenes": [
                {
                    "scene_index": 0,
                    "start_ms": 0,
                    "end_ms": 5000,
                    "duration_ms": 5000,
                },
                {
                    "scene_index": 1,
                    "start_ms": 5000,
                    "end_ms": 12500,
                    "duration_ms": 7500,
                },
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_006",
            task_type="scene_detection",
            video_id="video_001",
            ml_result=ml_result,
        )

        assert len(envelopes) == 2
        assert envelopes[0]["artifact_type"] == "scene.detection"
        assert envelopes[0]["span_start_ms"] == 0
        assert envelopes[0]["span_end_ms"] == 5000
        assert envelopes[1]["span_start_ms"] == 5000
        assert envelopes[1]["span_end_ms"] == 12500


class TestUnknownTaskType:
    """Tests for error handling with unknown task types."""

    def test_unknown_task_type(self):
        """Test that unknown task type raises ValueError."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "unknown",
            "producer_version": "1.0.0",
            "model_profile": "balanced",
            "results": [],
        }

        with pytest.raises(ValueError, match="Unknown task type"):
            ArtifactTransformer.transform_ml_result(
                task_id="task_007",
                task_type="unknown_task",
                video_id="video_001",
                ml_result=ml_result,
            )


class TestEnvelopeStructure:
    """Tests for ArtifactEnvelope structure and fields."""

    def test_envelope_has_required_fields(self):
        """Test that envelopes contain all required fields."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "bounding_box": {
                        "x": 100.0,
                        "y": 150.0,
                        "width": 200.0,
                        "height": 300.0,
                    },
                    "frame_number": 450,
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_001",
            task_type="object_detection",
            video_id="video_001",
            ml_result=ml_result,
        )

        envelope = envelopes[0]
        required_fields = [
            "asset_id",
            "artifact_type",
            "schema_version",
            "span_start_ms",
            "span_end_ms",
            "payload_json",
            "config_hash",
            "input_hash",
            "producer",
            "producer_version",
            "model_profile",
            "run_id",
        ]

        for field in required_fields:
            assert field in envelope, f"Missing required field: {field}"

    def test_payload_json_is_valid_json(self):
        """Test that payload_json can be serialized to JSON."""
        ml_result = {
            "config_hash": "config_abc123",
            "input_hash": "input_xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "bounding_box": {
                        "x": 100.0,
                        "y": 150.0,
                        "width": 200.0,
                        "height": 300.0,
                    },
                    "frame_number": 450,
                }
            ],
        }

        envelopes = ArtifactTransformer.transform_ml_result(
            task_id="task_001",
            task_type="object_detection",
            video_id="video_001",
            ml_result=ml_result,
        )

        # Should be able to serialize to JSON without error
        json_str = json.dumps(envelopes[0]["payload_json"])
        assert json_str is not None
        # Should be able to deserialize back
        deserialized = json.loads(json_str)
        assert deserialized["label"] == "person"
