"""Test JobProducer service."""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.job_producer import JobProducer


class TestJobProducerInitialization:
    """Test JobProducer initialization and lifecycle."""

    def test_initialization_with_default_redis_url(self):
        """Test JobProducer initializes with default Redis URL from config."""
        producer = JobProducer()
        assert producer.redis_url == "redis://valkey:6379/0"
        assert producer.pool is None

    def test_initialization_with_custom_redis_url(self):
        """Test JobProducer initializes with custom Redis URL."""
        custom_url = "redis://redis-server:6379"
        producer = JobProducer(redis_url=custom_url)
        assert producer.redis_url == custom_url

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self):
        """Test initialize() creates Redis connection pool."""
        producer = JobProducer()

        with patch("src.services.job_producer.create_pool") as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            await producer.initialize()

            assert producer.pool is not None
            mock_create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_pool(self):
        """Test close() closes Redis connection pool."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.close()

        producer.pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_pool(self):
        """Test close() handles None pool gracefully."""
        producer = JobProducer()
        producer.pool = None

        # Should not raise error
        await producer.close()


class TestJobProducerMLJobsEnqueueing:
    """Test ml_jobs queue enqueueing logic."""

    @pytest.mark.asyncio
    async def test_enqueue_task_object_detection(self):
        """Test enqueueing object detection task to ml_jobs queue."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        job_id = await producer.enqueue_task(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        assert job_id == "ml_task_123"
        producer.pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_task_transcription(self):
        """Test enqueueing transcription task to ml_jobs queue."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        job_id = await producer.enqueue_task(
            task_id="task_789",
            task_type="transcription",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        assert job_id == "ml_task_789"
        producer.pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_task_with_config(self):
        """Test enqueueing to ml_jobs with configuration."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        config = {"model": "yolov8n", "confidence": 0.5}

        await producer.enqueue_task(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
            config=config,
        )

        # Verify enqueue_job was called with config
        call_args = producer.pool.enqueue_job.call_args
        assert call_args[0][5] == config

    @pytest.mark.asyncio
    async def test_enqueue_task_without_pool_raises_error(self):
        """Test enqueueing to ml_jobs without initialized pool raises RuntimeError."""
        producer = JobProducer()
        producer.pool = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await producer.enqueue_task(
                task_id="task_123",
                task_type="object_detection",
                video_id="video_456",
                video_path="/path/to/video.mp4",
            )

    @pytest.mark.asyncio
    async def test_enqueue_task_unknown_type_raises_error(self):
        """Test enqueueing unknown task type to ml_jobs raises ValueError."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        with pytest.raises(ValueError, match="Unknown task type"):
            await producer.enqueue_task(
                task_id="task_123",
                task_type="unknown_task",
                video_id="video_456",
                video_path="/path/to/video.mp4",
            )

    @pytest.mark.asyncio
    async def test_enqueue_task_payload_structure(self):
        """Test ml_jobs payload has correct structure."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_task(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
            config={"model": "yolov8n"},
        )

        # Verify payload structure
        call_args = producer.pool.enqueue_job.call_args
        call_args_list = call_args[0]
        call_kwargs = call_args[1]

        # All function parameters should be passed as positional arguments
        assert call_args_list[0] == "process_ml_task"
        assert call_args_list[1] == "task_123"
        assert call_args_list[2] == "object_detection"
        assert call_args_list[3] == "video_456"
        assert call_args_list[4] == "/path/to/video.mp4"
        assert call_args_list[5] == {"model": "yolov8n"}

        # Only arq-specific parameters as kwargs
        assert call_kwargs["_job_id"] == "ml_task_123"
        assert call_kwargs["_queue_name"] == "ml_jobs"

    @pytest.mark.asyncio
    async def test_enqueue_task_queue_name(self):
        """Test ml_jobs enqueueing uses ml_jobs queue."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_task(
            task_id="task_123",
            task_type="transcription",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        # Verify queue name is 'ml_jobs'
        call_kwargs = producer.pool.enqueue_job.call_args[1]
        assert call_kwargs["_queue_name"] == "ml_jobs"

    @pytest.mark.asyncio
    async def test_enqueue_task_handler_name(self):
        """Test ml_jobs enqueueing uses process_ml_task handler."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_task(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        # Verify handler name is 'process_ml_task'
        call_args = producer.pool.enqueue_job.call_args
        assert call_args[0][0] == "process_ml_task"

    @pytest.mark.asyncio
    async def test_enqueue_task_job_id_format(self):
        """Test ml_jobs job_id follows ml_{task_id} format."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_task(
            task_id="abc-123-def",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        # Verify job_id format
        call_kwargs = producer.pool.enqueue_job.call_args[1]
        assert call_kwargs["_job_id"] == "ml_abc-123-def"

    @pytest.mark.asyncio
    async def test_enqueue_task_all_task_types(self):
        """Test all supported task types can be enqueued."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        task_types = [
            "object_detection",
            "face_detection",
            "transcription",
            "ocr",
            "place_detection",
            "scene_detection",
        ]

        for task_type in task_types:
            producer.pool.reset_mock()

            await producer.enqueue_task(
                task_id=f"task_{task_type}",
                task_type=task_type,
                video_id="video_456",
                video_path="/path/to/video.mp4",
            )

            producer.pool.enqueue_job.assert_called_once()
