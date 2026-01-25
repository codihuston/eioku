"""Test JobProducer service."""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.job_producer import JobProducer


class TestJobProducerQueueRouting:
    """Test queue routing logic."""

    def test_all_tasks_route_to_single_jobs_queue(self):
        """Test all task types route to single 'jobs' queue."""
        producer = JobProducer()

        all_tasks = [
            "object_detection",
            "face_detection",
            "place_detection",
            "scene_detection",
            "transcription",
            "ocr",
        ]

        for task_type in all_tasks:
            queue_name = producer._get_queue_name(task_type)
            assert queue_name == "jobs", f"{task_type} should route to 'jobs' queue"

    def test_unknown_task_type_raises_error(self):
        """Test unknown task type raises ValueError."""
        producer = JobProducer()

        with pytest.raises(ValueError, match="Unknown task type"):
            producer._get_queue_name("unknown_task_type")

    def test_unknown_task_type_error_message(self):
        """Test error message includes task type."""
        producer = JobProducer()

        with pytest.raises(ValueError) as exc_info:
            producer._get_queue_name("invalid_task")

        assert "invalid_task" in str(exc_info.value)


class TestJobProducerWorkerCapabilities:
    """Test worker capability checking."""

    def test_gpu_required_tasks_need_gpu(self):
        """Test GPU-required tasks need GPU available."""
        producer = JobProducer()

        gpu_tasks = [
            "object_detection",
            "face_detection",
            "place_detection",
            "scene_detection",
        ]

        for task_type in gpu_tasks:
            # Should not handle without GPU
            assert not producer.can_worker_handle(task_type, gpu_available=False)
            # Should handle with GPU
            assert producer.can_worker_handle(task_type, gpu_available=True)

    def test_cpu_capable_tasks_work_on_any_worker(self):
        """Test CPU-capable tasks work on any worker."""
        producer = JobProducer()

        cpu_tasks = ["transcription", "ocr"]

        for task_type in cpu_tasks:
            # Should handle without GPU
            assert producer.can_worker_handle(task_type, gpu_available=False)
            # Should also handle with GPU
            assert producer.can_worker_handle(task_type, gpu_available=True)

    def test_unknown_task_cannot_be_handled(self):
        """Test unknown task type cannot be handled."""
        producer = JobProducer()

        assert not producer.can_worker_handle("unknown_task", gpu_available=True)
        assert not producer.can_worker_handle("unknown_task", gpu_available=False)


class TestJobProducerEnqueueing:
    """Test job enqueueing logic."""

    @pytest.mark.asyncio
    async def test_enqueue_task_gpu_job(self):
        """Test enqueueing GPU task to single jobs queue."""
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
    async def test_enqueue_task_cpu_job(self):
        """Test enqueueing CPU-capable task to single jobs queue."""
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
        """Test enqueueing task with configuration."""
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
        assert call_args[1]["config"] == config

    @pytest.mark.asyncio
    async def test_enqueue_task_without_pool_raises_error(self):
        """Test enqueueing without initialized pool raises RuntimeError."""
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
        """Test enqueueing unknown task type raises ValueError."""
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
        """Test job payload has correct structure."""
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
        call_kwargs = producer.pool.enqueue_job.call_args[1]
        assert call_kwargs["task_id"] == "task_123"
        assert call_kwargs["task_type"] == "object_detection"
        assert call_kwargs["video_id"] == "video_456"
        assert call_kwargs["video_path"] == "/path/to/video.mp4"
        assert call_kwargs["config"] == {"model": "yolov8n"}
        assert call_kwargs["job_id"] == "ml_task_123"
        assert call_kwargs["_queue_name"] == "jobs"

    @pytest.mark.asyncio
    async def test_enqueue_task_cpu_capable_queue_name(self):
        """Test CPU-capable task uses single jobs queue."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_task(
            task_id="task_123",
            task_type="transcription",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        # Verify queue name is 'jobs' (single queue for all tasks)
        call_kwargs = producer.pool.enqueue_job.call_args[1]
        assert call_kwargs["_queue_name"] == "jobs"


class TestJobProducerMLJobsEnqueueing:
    """Test ml_jobs queue enqueueing logic."""

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_gpu_task(self):
        """Test enqueueing GPU task to ml_jobs queue."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        job_id = await producer.enqueue_to_ml_jobs(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        assert job_id == "ml_task_123"
        producer.pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_cpu_task(self):
        """Test enqueueing CPU-capable task to ml_jobs queue."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        job_id = await producer.enqueue_to_ml_jobs(
            task_id="task_789",
            task_type="transcription",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        assert job_id == "ml_task_789"
        producer.pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_with_config(self):
        """Test enqueueing to ml_jobs with configuration."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        config = {"model": "yolov8n", "confidence": 0.5}

        await producer.enqueue_to_ml_jobs(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
            config=config,
        )

        # Verify enqueue_job was called with config
        call_args = producer.pool.enqueue_job.call_args
        assert call_args[1]["config"] == config

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_without_pool_raises_error(self):
        """Test enqueueing to ml_jobs without initialized pool raises RuntimeError."""
        producer = JobProducer()
        producer.pool = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await producer.enqueue_to_ml_jobs(
                task_id="task_123",
                task_type="object_detection",
                video_id="video_456",
                video_path="/path/to/video.mp4",
            )

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_unknown_type_raises_error(self):
        """Test enqueueing unknown task type to ml_jobs raises ValueError."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        with pytest.raises(ValueError, match="Unknown task type"):
            await producer.enqueue_to_ml_jobs(
                task_id="task_123",
                task_type="unknown_task",
                video_id="video_456",
                video_path="/path/to/video.mp4",
            )

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_payload_structure(self):
        """Test ml_jobs payload has correct structure."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_to_ml_jobs(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
            config={"model": "yolov8n"},
        )

        # Verify payload structure
        call_kwargs = producer.pool.enqueue_job.call_args[1]
        assert call_kwargs["task_id"] == "task_123"
        assert call_kwargs["task_type"] == "object_detection"
        assert call_kwargs["video_id"] == "video_456"
        assert call_kwargs["video_path"] == "/path/to/video.mp4"
        assert call_kwargs["config"] == {"model": "yolov8n"}
        assert call_kwargs["job_id"] == "ml_task_123"
        assert call_kwargs["_queue_name"] == "ml_jobs"

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_queue_name(self):
        """Test ml_jobs enqueueing uses ml_jobs queue."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_to_ml_jobs(
            task_id="task_123",
            task_type="transcription",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        # Verify queue name is 'ml_jobs'
        call_kwargs = producer.pool.enqueue_job.call_args[1]
        assert call_kwargs["_queue_name"] == "ml_jobs"

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_handler_name(self):
        """Test ml_jobs enqueueing uses process_inference_job handler."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_to_ml_jobs(
            task_id="task_123",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        # Verify handler name is 'process_inference_job'
        call_args = producer.pool.enqueue_job.call_args
        assert call_args[0][0] == "process_inference_job"

    @pytest.mark.asyncio
    async def test_enqueue_to_ml_jobs_job_id_format(self):
        """Test ml_jobs job_id follows ml_{task_id} format."""
        producer = JobProducer()
        producer.pool = AsyncMock()

        await producer.enqueue_to_ml_jobs(
            task_id="abc-123-def",
            task_type="object_detection",
            video_id="video_456",
            video_path="/path/to/video.mp4",
        )

        # Verify job_id format
        call_kwargs = producer.pool.enqueue_job.call_args[1]
        assert call_kwargs["job_id"] == "ml_abc-123-def"


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


class TestJobProducerTaskTypes:
    """Test task type classification."""

    def test_gpu_required_tasks_constant(self):
        """Test GPU_REQUIRED_TASKS constant contains expected task types."""
        expected_gpu_tasks = {
            "object_detection",
            "face_detection",
            "place_detection",
            "scene_detection",
        }
        assert JobProducer.GPU_REQUIRED_TASKS == expected_gpu_tasks

    def test_cpu_capable_tasks_constant(self):
        """Test CPU_CAPABLE_TASKS constant contains expected task types."""
        expected_cpu_tasks = {"transcription", "ocr"}
        assert JobProducer.CPU_CAPABLE_TASKS == expected_cpu_tasks

    def test_supported_tasks_includes_all(self):
        """Test SUPPORTED_TASKS includes both GPU and CPU-capable tasks."""
        expected_all = {
            "object_detection",
            "face_detection",
            "place_detection",
            "scene_detection",
            "transcription",
            "ocr",
        }
        assert JobProducer.SUPPORTED_TASKS == expected_all

    def test_no_overlap_between_gpu_and_cpu_tasks(self):
        """Test GPU and CPU-capable task sets don't overlap."""
        overlap = JobProducer.GPU_REQUIRED_TASKS & JobProducer.CPU_CAPABLE_TASKS
        assert len(overlap) == 0, "GPU and CPU-capable task sets should not overlap"
