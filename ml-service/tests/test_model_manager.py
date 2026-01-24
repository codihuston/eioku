"""Tests for ModelManager."""

import pytest

from src.services.model_manager import ModelManager


@pytest.fixture
def model_manager():
    """Create model manager instance."""
    return ModelManager(cache_dir="/tmp/test_models")


def test_model_manager_init(model_manager):
    """Test model manager initialization."""
    assert model_manager is not None
    assert model_manager.cache_dir.exists()


def test_gpu_detection(model_manager):
    """Test GPU detection."""
    gpu_available = model_manager.detect_gpu()
    assert isinstance(gpu_available, bool)


def test_get_gpu_info(model_manager):
    """Test GPU info retrieval."""
    gpu_info = model_manager.get_gpu_info()
    assert "gpu_available" in gpu_info
    assert "gpu_device_name" in gpu_info
    assert "gpu_memory_total_mb" in gpu_info
    assert "gpu_memory_used_mb" in gpu_info


def test_get_device(model_manager):
    """Test device string generation."""
    device = model_manager._get_device()
    assert device in ["cuda", "cpu"]
