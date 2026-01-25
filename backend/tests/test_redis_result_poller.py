"""Unit tests for Redis result polling.

Tests the RedisResultPoller class which polls Redis for ML Service results
with exponential backoff.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.workers.redis_result_poller import RedisResultPoller


@pytest.fixture
def poller():
    """Create a RedisResultPoller instance for testing."""
    return RedisResultPoller(redis_url="redis://localhost:6379")


class TestRedisResultPollerConnection:
    """Tests for Redis connection management."""

    @pytest.mark.asyncio
    async def test_connect_establishes_connection(self, poller):
        """Test that connect() establishes Redis connection."""
        mock_client = AsyncMock()

        async def mock_from_url(url):
            return mock_client

        with patch(
            "src.workers.redis_result_poller.redis.from_url", side_effect=mock_from_url
        ):
            await poller.connect()

            assert poller.redis_client is not None

    @pytest.mark.asyncio
    async def test_close_closes_connection(self, poller):
        """Test that close() closes Redis connection."""
        poller.redis_client = AsyncMock()

        await poller.close()

        poller.redis_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_none_client(self, poller):
        """Test that close() handles None client gracefully."""
        poller.redis_client = None

        # Should not raise
        await poller.close()


class TestRedisResultPolling:
    """Tests for result polling functionality."""

    @pytest.mark.asyncio
    async def test_poll_for_result_immediate_success(self, poller):
        """Test polling when result is immediately available."""
        result_data = {
            "config_hash": "abc123",
            "input_hash": "xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "detections": [],
        }

        poller.redis_client = AsyncMock()
        poller.redis_client.get.return_value = json.dumps(result_data).encode()

        result = await poller.poll_for_result(task_id="task_001")

        assert result == result_data
        poller.redis_client.get.assert_called_once_with("ml_result:task_001")

    @pytest.mark.asyncio
    async def test_poll_for_result_with_retries(self, poller):
        """Test polling with multiple retries before success."""
        result_data = {
            "config_hash": "abc123",
            "input_hash": "xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
            "detections": [],
        }

        poller.redis_client = AsyncMock()
        # First two calls return None, third returns result
        poller.redis_client.get.side_effect = [
            None,
            None,
            json.dumps(result_data).encode(),
        ]

        result = await poller.poll_for_result(
            task_id="task_001",
            initial_delay=0.01,
            max_delay=0.1,
            timeout=10.0,
        )

        assert result == result_data
        assert poller.redis_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_poll_for_result_timeout(self, poller):
        """Test polling timeout when result never arrives."""
        poller.redis_client = AsyncMock()
        poller.redis_client.get.return_value = None

        with pytest.raises(TimeoutError, match="polling timeout exceeded"):
            await poller.poll_for_result(
                task_id="task_001",
                initial_delay=0.01,
                max_delay=0.02,
                timeout=0.05,
            )

    @pytest.mark.asyncio
    async def test_poll_for_result_invalid_json(self, poller):
        """Test polling with invalid JSON in Redis."""
        poller.redis_client = AsyncMock()
        poller.redis_client.get.return_value = b"invalid json {{"

        with pytest.raises(ValueError, match="Invalid JSON"):
            await poller.poll_for_result(task_id="task_001")

    @pytest.mark.asyncio
    async def test_poll_for_result_not_connected(self, poller):
        """Test polling raises error when not connected."""
        poller.redis_client = None

        with pytest.raises(RuntimeError, match="not connected"):
            await poller.poll_for_result(task_id="task_001")

    @pytest.mark.asyncio
    async def test_poll_for_result_exponential_backoff(self, poller):
        """Test that polling uses exponential backoff."""
        result_data = {"config_hash": "abc123"}

        poller.redis_client = AsyncMock()
        # Return None for first 5 calls, then result
        poller.redis_client.get.side_effect = [
            None,
            None,
            None,
            None,
            None,
            json.dumps(result_data).encode(),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await poller.poll_for_result(
                task_id="task_001",
                initial_delay=1.0,
                max_delay=30.0,
                timeout=1000.0,
            )

            assert result == result_data
            # Check that sleep was called with exponential backoff
            # 1.0, 2.0, 4.0, 8.0, 16.0
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls == [1.0, 2.0, 4.0, 8.0, 16.0]

    @pytest.mark.asyncio
    async def test_poll_for_result_backoff_capped(self, poller):
        """Test that exponential backoff is capped at max_delay."""
        result_data = {"config_hash": "abc123"}

        poller.redis_client = AsyncMock()
        # Return None for many calls to trigger backoff capping
        poller.redis_client.get.side_effect = [None] * 10 + [
            json.dumps(result_data).encode()
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await poller.poll_for_result(
                task_id="task_001",
                initial_delay=1.0,
                max_delay=5.0,
                timeout=1000.0,
            )

            assert result == result_data
            # Check that sleep never exceeds max_delay
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert all(delay <= 5.0 for delay in sleep_calls)
            # Last calls should be at max_delay
            assert sleep_calls[-1] == 5.0

    @pytest.mark.asyncio
    async def test_poll_for_result_continues_on_redis_error(self, poller):
        """Test that polling continues on Redis errors."""
        result_data = {"config_hash": "abc123"}

        poller.redis_client = AsyncMock()
        # First call raises error, subsequent calls return result
        poller.redis_client.get.side_effect = [
            Exception("Redis error"),
            None,
            json.dumps(result_data).encode(),
        ]

        result = await poller.poll_for_result(
            task_id="task_001",
            initial_delay=0.01,
            max_delay=0.1,
            timeout=10.0,
        )

        assert result == result_data
        assert poller.redis_client.get.call_count == 3


class TestRedisResultDeletion:
    """Tests for result deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_result_success(self, poller):
        """Test successful result deletion."""
        poller.redis_client = AsyncMock()
        poller.redis_client.delete.return_value = 1

        deleted = await poller.delete_result(task_id="task_001")

        assert deleted is True
        poller.redis_client.delete.assert_called_once_with("ml_result:task_001")

    @pytest.mark.asyncio
    async def test_delete_result_not_found(self, poller):
        """Test deletion when result key doesn't exist."""
        poller.redis_client = AsyncMock()
        poller.redis_client.delete.return_value = 0

        deleted = await poller.delete_result(task_id="task_001")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_result_not_connected(self, poller):
        """Test deletion raises error when not connected."""
        poller.redis_client = None

        with pytest.raises(RuntimeError, match="not connected"):
            await poller.delete_result(task_id="task_001")

    @pytest.mark.asyncio
    async def test_delete_result_redis_error(self, poller):
        """Test deletion error handling."""
        poller.redis_client = AsyncMock()
        poller.redis_client.delete.side_effect = Exception("Redis error")

        with pytest.raises(Exception, match="Redis error"):
            await poller.delete_result(task_id="task_001")


class TestRedisResultExistenceCheck:
    """Tests for result existence checking."""

    @pytest.mark.asyncio
    async def test_check_result_exists_true(self, poller):
        """Test checking when result exists."""
        poller.redis_client = AsyncMock()
        poller.redis_client.exists.return_value = 1

        exists = await poller.check_result_exists(task_id="task_001")

        assert exists is True
        poller.redis_client.exists.assert_called_once_with("ml_result:task_001")

    @pytest.mark.asyncio
    async def test_check_result_exists_false(self, poller):
        """Test checking when result doesn't exist."""
        poller.redis_client = AsyncMock()
        poller.redis_client.exists.return_value = 0

        exists = await poller.check_result_exists(task_id="task_001")

        assert exists is False

    @pytest.mark.asyncio
    async def test_check_result_exists_not_connected(self, poller):
        """Test existence check raises error when not connected."""
        poller.redis_client = None

        with pytest.raises(RuntimeError, match="not connected"):
            await poller.check_result_exists(task_id="task_001")

    @pytest.mark.asyncio
    async def test_check_result_exists_redis_error(self, poller):
        """Test existence check error handling."""
        poller.redis_client = AsyncMock()
        poller.redis_client.exists.side_effect = Exception("Redis error")

        with pytest.raises(Exception, match="Redis error"):
            await poller.check_result_exists(task_id="task_001")


class TestRedisResultPollerIntegration:
    """Integration tests for complete polling workflow."""

    @pytest.mark.asyncio
    async def test_complete_polling_workflow(self, poller):
        """Test complete workflow: connect, poll, delete, close."""
        result_data = {
            "config_hash": "abc123",
            "input_hash": "xyz789",
            "run_id": "run_001",
            "producer": "yolo",
            "producer_version": "8.0.0",
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

        mock_client = AsyncMock()

        async def mock_from_url(url):
            return mock_client

        with patch(
            "src.workers.redis_result_poller.redis.from_url", side_effect=mock_from_url
        ):
            # Setup mock responses
            mock_client.get.return_value = json.dumps(result_data).encode()
            mock_client.delete.return_value = 1

            # Connect
            await poller.connect()
            assert poller.redis_client is not None

            # Poll for result
            result = await poller.poll_for_result(task_id="task_001")
            assert result == result_data

            # Delete result
            deleted = await poller.delete_result(task_id="task_001")
            assert deleted is True

            # Close
            await poller.close()
            mock_client.close.assert_called_once()
