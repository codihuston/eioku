"""Unit tests for ML Service job handler."""

from unittest.mock import MagicMock, patch

import pytest

from src.workers.job_handler import (
    TASK_TYPE_TO_ENDPOINT,
    _execute_inference,
    _persist_artifacts,
    _transform_to_artifacts,
    process_inference_job,
)


class TestTaskTypeMapping:
    """Test task type to endpoint mapping."""

    def test_all_task_types_mapped(self):
        """Test that all task types have endpoint mappings."""
        expected_types = {
            "object_detection",
            "face_detection",
            "transcription",
            "ocr",
            "place_detection",
            "scene_detection",
        }
        assert set(TASK_TYPE_TO_ENDPOINT.keys()) == expected_types

    def test_endpoint_names_valid(self):
        """Test that endpoint names are valid."""
        valid_endpoints = {
            "objects",
            "faces",
            "transcribe",
            "ocr",
            "places",
            "scenes",
        }
        assert set(TASK_TYPE_TO_ENDPOINT.values()) == valid_endpoints


class TestTransformToArtifacts:
    """Test artifact transformation logic."""

    def test_transform_object_detection_response(self):
        """Test transforming object detection response."""
        ml_response = {
            "run_id": "run-123",
            "config_hash": "config-hash-123",
            "input_hash": "input-hash-123",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "frame_index": 0,
                    "timestamp_ms": 0,
                    "label": "person",
                    "confidence": 0.95,
                    "bbox": {"x": 10, "y": 20, "width": 100, "height": 200},
                    "start_ms": 0,
                    "end_ms": 33,
                },
                {
                    "frame_index": 1,
                    "timestamp_ms": 33,
                    "label": "car",
                    "confidence": 0.87,
                    "bbox": {"x": 50, "y": 60, "width": 150, "height": 100},
                    "start_ms": 33,
                    "end_ms": 66,
                },
            ],
        }

        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id="task-123",
            video_id="video-123",
            task_type="object_detection",
        )

        assert len(artifacts) == 2
        assert artifacts[0]["task_id"] == "task-123"
        assert artifacts[0]["video_id"] == "video-123"
        assert artifacts[0]["task_type"] == "object_detection"
        assert artifacts[0]["span_start_ms"] == 0
        assert artifacts[0]["span_end_ms"] == 33
        assert artifacts[0]["config_hash"] == "config-hash-123"
        assert artifacts[0]["run_id"] == "run-123"

    def test_transform_transcription_response(self):
        """Test transforming transcription response."""
        ml_response = {
            "run_id": "run-456",
            "config_hash": "config-hash-456",
            "input_hash": "input-hash-456",
            "producer": "whisper",
            "producer_version": "3.0",
            "model_profile": "balanced",
            "detections": [
                {
                    "start_ms": 0,
                    "end_ms": 5000,
                    "text": "Hello world",
                    "confidence": 0.98,
                },
                {
                    "start_ms": 5000,
                    "end_ms": 10000,
                    "text": "This is a test",
                    "confidence": 0.95,
                },
            ],
        }

        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id="task-456",
            video_id="video-456",
            task_type="transcription",
        )

        assert len(artifacts) == 2
        assert artifacts[0]["span_start_ms"] == 0
        assert artifacts[0]["span_end_ms"] == 5000
        assert artifacts[1]["span_start_ms"] == 5000
        assert artifacts[1]["span_end_ms"] == 10000

    def test_transform_scene_detection_response(self):
        """Test transforming scene detection response."""
        ml_response = {
            "run_id": "run-789",
            "config_hash": "config-hash-789",
            "input_hash": "input-hash-789",
            "producer": "scenedetect",
            "producer_version": "0.6.0",
            "model_profile": "balanced",
            "scenes": [
                {"scene_index": 0, "start_ms": 0, "end_ms": 5000},
                {"scene_index": 1, "start_ms": 5000, "end_ms": 10000},
            ],
        }

        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id="task-789",
            video_id="video-789",
            task_type="scene_detection",
        )

        assert len(artifacts) == 2
        assert artifacts[0]["span_start_ms"] == 0
        assert artifacts[0]["span_end_ms"] == 5000

    def test_transform_empty_detections(self):
        """Test transforming response with no detections."""
        ml_response = {
            "run_id": "run-empty",
            "config_hash": "config-hash-empty",
            "input_hash": "input-hash-empty",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "detections": [],
        }

        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id="task-empty",
            video_id="video-empty",
            task_type="object_detection",
        )

        assert len(artifacts) == 0

    def test_transform_invalid_time_span(self):
        """Test that invalid time spans are skipped."""
        ml_response = {
            "run_id": "run-invalid",
            "config_hash": "config-hash-invalid",
            "input_hash": "input-hash-invalid",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "detections": [
                {
                    "frame_index": 0,
                    "timestamp_ms": 0,
                    "label": "person",
                    "confidence": 0.95,
                    "start_ms": 100,
                    "end_ms": 50,  # Invalid: start > end
                },
                {
                    "frame_index": 1,
                    "timestamp_ms": 33,
                    "label": "car",
                    "confidence": 0.87,
                    "start_ms": 50,
                    "end_ms": 100,  # Valid
                },
            ],
        }

        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id="task-invalid",
            video_id="video-invalid",
            task_type="object_detection",
        )

        # Only the valid detection should be included
        assert len(artifacts) == 1
        assert artifacts[0]["span_start_ms"] == 50
        assert artifacts[0]["span_end_ms"] == 100

    def test_transform_negative_time_span(self):
        """Test that negative time spans are skipped."""
        ml_response = {
            "run_id": "run-negative",
            "config_hash": "config-hash-negative",
            "input_hash": "input-hash-negative",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "detections": [
                {
                    "frame_index": 0,
                    "timestamp_ms": 0,
                    "label": "person",
                    "confidence": 0.95,
                    "start_ms": -100,  # Invalid: negative
                    "end_ms": 50,
                },
                {
                    "frame_index": 1,
                    "timestamp_ms": 33,
                    "label": "car",
                    "confidence": 0.87,
                    "start_ms": 50,
                    "end_ms": 100,  # Valid
                },
            ],
        }

        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id="task-negative",
            video_id="video-negative",
            task_type="object_detection",
        )

        # Only the valid detection should be included
        assert len(artifacts) == 1

    def test_transform_empty_response(self):
        """Test that empty response raises ValueError."""
        with pytest.raises(ValueError, match="ml_response cannot be empty"):
            _transform_to_artifacts(
                ml_response=None,
                task_id="task-none",
                video_id="video-none",
                task_type="object_detection",
            )

    def test_transform_preserves_provenance(self):
        """Test that provenance metadata is preserved."""
        ml_response = {
            "run_id": "run-prov",
            "config_hash": "config-hash-prov",
            "input_hash": "input-hash-prov",
            "producer": "custom-producer",
            "producer_version": "2.0.0",
            "model_profile": "fast",
            "detections": [
                {
                    "frame_index": 0,
                    "timestamp_ms": 0,
                    "label": "person",
                    "confidence": 0.95,
                    "start_ms": 0,
                    "end_ms": 33,
                },
            ],
        }

        artifacts = _transform_to_artifacts(
            ml_response=ml_response,
            task_id="task-prov",
            video_id="video-prov",
            task_type="object_detection",
        )

        assert len(artifacts) == 1
        assert artifacts[0]["config_hash"] == "config-hash-prov"
        assert artifacts[0]["input_hash"] == "input-hash-prov"
        assert artifacts[0]["run_id"] == "run-prov"
        assert artifacts[0]["producer"] == "custom-producer"
        assert artifacts[0]["producer_version"] == "2.0.0"
        assert artifacts[0]["model_profile"] == "fast"


class TestPersistArtifacts:
    """Test artifact persistence logic."""

    @pytest.mark.asyncio
    async def test_persist_artifacts_success(self):
        """Test successful artifact persistence."""
        artifacts = [
            {
                "task_id": "task-123",
                "video_id": "video-123",
                "task_type": "object_detection",
                "span_start_ms": 0,
                "span_end_ms": 33,
                "payload": {"label": "person", "confidence": 0.95},
                "config_hash": "config-hash",
                "input_hash": "input-hash",
                "run_id": "run-123",
                "producer": "yolo",
                "producer_version": "8.0.0",
                "model_profile": "balanced",
            },
        ]

        with patch("psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            count = await _persist_artifacts(
                task_id="task-123",
                video_id="video-123",
                task_type="object_detection",
                artifacts=artifacts,
            )

            assert count == 1
            mock_cursor.execute.assert_called()
            mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_empty_artifacts(self):
        """Test persisting empty artifact list."""
        count = await _persist_artifacts(
            task_id="task-empty",
            video_id="video-empty",
            task_type="object_detection",
            artifacts=[],
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_persist_artifacts_database_error(self):
        """Test handling of database errors."""
        artifacts = [
            {
                "task_id": "task-error",
                "video_id": "video-error",
                "task_type": "object_detection",
                "span_start_ms": 0,
                "span_end_ms": 33,
                "payload": {"label": "person"},
                "config_hash": "config-hash",
                "input_hash": "input-hash",
                "run_id": "run-error",
                "producer": "yolo",
                "producer_version": "8.0.0",
                "model_profile": "balanced",
            },
        ]

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = Exception("Database connection failed")

            with pytest.raises(RuntimeError, match="Failed to persist artifacts"):
                await _persist_artifacts(
                    task_id="task-error",
                    video_id="video-error",
                    task_type="object_detection",
                    artifacts=artifacts,
                )


class TestExecuteInference:
    """Test inference execution logic."""

    @pytest.mark.asyncio
    async def test_execute_inference_unknown_task_type(self):
        """Test that unknown task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            await _execute_inference(
                task_type="unknown_task",
                video_path="/path/to/video.mp4",
                config={},
            )

    @pytest.mark.asyncio
    async def test_execute_inference_object_detection(self):
        """Test object detection inference execution."""
        with patch("src.api.inference.detect_objects") as mock_detect:
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {
                "run_id": "run-123",
                "config_hash": "config-hash",
                "input_hash": "input-hash",
                "detections": [],
            }
            mock_detect.return_value = mock_response

            result = await _execute_inference(
                task_type="object_detection",
                video_path="/path/to/video.mp4",
                config={"model_name": "yolov8n.pt"},
            )

            assert result["run_id"] == "run-123"
            mock_detect.assert_called_once()


class TestProcessInferenceJob:
    """Test main job handler logic."""

    @pytest.mark.asyncio
    async def test_process_inference_job_unknown_task_type(self):
        """Test that unknown task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            await process_inference_job(
                task_id="task-123",
                task_type="unknown_task",
                video_id="video-123",
                video_path="/path/to/video.mp4",
            )

    @pytest.mark.asyncio
    async def test_process_inference_job_success(self):
        """Test successful job processing."""
        with patch("src.workers.job_handler._execute_inference") as mock_infer, patch(
            "src.workers.job_handler._persist_artifacts"
        ) as mock_persist:
            mock_infer.return_value = {
                "run_id": "run-123",
                "config_hash": "config-hash",
                "input_hash": "input-hash",
                "detections": [
                    {
                        "frame_index": 0,
                        "timestamp_ms": 0,
                        "label": "person",
                        "confidence": 0.95,
                        "start_ms": 0,
                        "end_ms": 33,
                    },
                ],
            }
            mock_persist.return_value = 1

            result = await process_inference_job(
                task_id="task-123",
                task_type="object_detection",
                video_id="video-123",
                video_path="/path/to/video.mp4",
                config={"model_name": "yolov8n.pt"},
            )

            assert result["task_id"] == "task-123"
            assert result["status"] == "completed"
            assert result["artifact_count"] == 1
            mock_infer.assert_called_once()
            mock_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_inference_job_inference_failure(self):
        """Test handling of inference failure."""
        with patch("src.workers.job_handler._execute_inference") as mock_infer:
            mock_infer.side_effect = RuntimeError("Inference failed")

            with pytest.raises(RuntimeError, match="Inference failed"):
                await process_inference_job(
                    task_id="task-123",
                    task_type="object_detection",
                    video_id="video-123",
                    video_path="/path/to/video.mp4",
                )

    @pytest.mark.asyncio
    async def test_process_inference_job_persistence_failure(self):
        """Test handling of persistence failure."""
        with patch("src.workers.job_handler._execute_inference") as mock_infer, patch(
            "src.workers.job_handler._persist_artifacts"
        ) as mock_persist:
            mock_infer.return_value = {
                "run_id": "run-123",
                "config_hash": "config-hash",
                "input_hash": "input-hash",
                "detections": [
                    {
                        "frame_index": 0,
                        "timestamp_ms": 0,
                        "label": "person",
                        "confidence": 0.95,
                        "start_ms": 0,
                        "end_ms": 33,
                    },
                ],
            }
            mock_persist.side_effect = RuntimeError("Database error")

            with pytest.raises(RuntimeError, match="Database error"):
                await process_inference_job(
                    task_id="task-123",
                    task_type="object_detection",
                    video_id="video-123",
                    video_path="/path/to/video.mp4",
                )
