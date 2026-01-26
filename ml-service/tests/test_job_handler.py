"""Unit tests for ML Service job handler."""

from unittest.mock import AsyncMock, patch

import pytest

from src.workers.job_handler import TASK_TYPE_TO_ENDPOINT, process_inference_job


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
                input_hash="abc123def456",
            )

    @pytest.mark.asyncio
    async def test_process_inference_job_object_detection(self):
        """Test successful object detection job processing."""
        with patch("src.workers.job_handler._execute_inference") as mock_infer:
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

            mock_redis_client = AsyncMock()
            with patch("src.workers.job_handler.redis") as mock_redis_module:
                mock_redis_module.from_url = AsyncMock(return_value=mock_redis_client)

                result = await process_inference_job(
                    task_id="task-123",
                    task_type="object_detection",
                    video_id="video-123",
                    video_path="/path/to/video.mp4",
                    input_hash="abc123def456",
                    config={"model_name": "yolov8n.pt"},
                )

                assert result["task_id"] == "task-123"
                assert result["status"] == "completed"
                assert result["result_key"] == "ml_result:task-123"

                # Verify Redis RPUSH was called
                mock_redis_client.rpush.assert_called_once()
                mock_redis_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_inference_job_transcription(self):
        """Test successful transcription job processing."""
        with patch("src.workers.job_handler._execute_inference") as mock_infer:
            mock_infer.return_value = {
                "run_id": "run-456",
                "config_hash": "config-hash",
                "input_hash": "input-hash",
                "detections": [
                    {
                        "start_ms": 0,
                        "end_ms": 5000,
                        "text": "Hello world",
                        "confidence": 0.98,
                    },
                ],
            }

            mock_redis_client = AsyncMock()
            with patch("src.workers.job_handler.redis") as mock_redis_module:
                mock_redis_module.from_url = AsyncMock(return_value=mock_redis_client)

                result = await process_inference_job(
                    task_id="task-456",
                    task_type="transcription",
                    video_id="video-456",
                    video_path="/path/to/video.mp4",
                    input_hash="xyz789abc123",
                )

                assert result["task_id"] == "task-456"
                assert result["status"] == "completed"
                mock_redis_client.rpush.assert_called_once()

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
                    input_hash="abc123def456",
                )

    @pytest.mark.asyncio
    async def test_process_inference_job_redis_failure(self):
        """Test handling of Redis failure."""
        with patch("src.workers.job_handler._execute_inference") as mock_infer:
            mock_infer.return_value = {
                "run_id": "run-123",
                "config_hash": "config-hash",
                "input_hash": "input-hash",
                "detections": [],
            }

            mock_redis_client = AsyncMock()
            mock_redis_client.rpush.side_effect = RuntimeError(
                "Redis connection failed"
            )
            with patch("src.workers.job_handler.redis") as mock_redis_module:
                mock_redis_module.from_url = AsyncMock(return_value=mock_redis_client)

                with pytest.raises(RuntimeError, match="Redis connection failed"):
                    await process_inference_job(
                        task_id="task-123",
                        task_type="object_detection",
                        video_id="video-123",
                        video_path="/path/to/video.mp4",
                        input_hash="abc123def456",
                    )

                # Verify Redis connection was closed even on error
                mock_redis_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_inference_job_all_task_types(self):
        """Test all supported task types can be processed."""
        task_types = [
            "object_detection",
            "face_detection",
            "transcription",
            "ocr",
            "place_detection",
            "scene_detection",
        ]

        for task_type in task_types:
            with patch("src.workers.job_handler._execute_inference") as mock_infer:
                mock_infer.return_value = {
                    "run_id": f"run-{task_type}",
                    "config_hash": "config-hash",
                    "input_hash": "input-hash",
                    "detections": [],
                }

                mock_redis_client = AsyncMock()
                with patch("src.workers.job_handler.redis") as mock_redis_module:
                    mock_redis_module.from_url = AsyncMock(
                        return_value=mock_redis_client
                    )

                    result = await process_inference_job(
                        task_id=f"task-{task_type}",
                        task_type=task_type,
                        video_id="video-123",
                        video_path="/path/to/video.mp4",
                        input_hash="abc123def456",
                    )

                    assert result["status"] == "completed"
                    mock_infer.assert_called_once()
                    mock_redis_client.rpush.assert_called_once()
