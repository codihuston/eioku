"""Tests for ML Service application."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Eioku ML Service"


def test_health_endpoint_exists(client):
    """Test health endpoint exists."""
    response = client.get("/health")
    # May fail if models not initialized, but endpoint should exist
    assert response.status_code in [200, 500]


def test_health_response_structure(client):
    """Test health response has correct structure."""
    from src.api import health
    from src.services.model_manager import ModelManager

    # Mock the globals
    health.MODELS_REGISTRY = {
        "yolov8n.pt": {"status": "ready", "type": "yolo"},
        "large-v3": {"status": "ready", "type": "whisper"},
    }
    health.MODEL_MANAGER = ModelManager()

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "gpu_available" in data
    assert "gpu_device_name" in data
    assert "gpu_memory_total_mb" in data
    assert "gpu_memory_used_mb" in data


def test_health_healthy_status(client):
    """Test health endpoint returns healthy status when all models ready."""
    from src.api import health
    from src.services.model_manager import ModelManager

    # Mock all models as ready
    health.MODELS_REGISTRY = {
        "yolov8n.pt": {"status": "ready", "type": "yolo"},
        "yolov8n-face.pt": {"status": "ready", "type": "yolo"},
        "large-v3": {"status": "ready", "type": "whisper"},
        "english": {"status": "ready", "type": "easyocr"},
    }
    health.MODEL_MANAGER = ModelManager()

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["models_loaded"]) == 4


def test_health_degraded_status(client):
    """Test health endpoint returns degraded status when some models fail."""
    from src.api import health
    from src.services.model_manager import ModelManager

    # Mock some models as failed
    health.MODELS_REGISTRY = {
        "yolov8n.pt": {"status": "ready", "type": "yolo"},
        "large-v3": {"status": "failed", "type": "whisper", "error": "Download failed"},
    }
    health.MODEL_MANAGER = ModelManager()

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "degraded"
    assert len(data["models_loaded"]) == 1


def test_health_unhealthy_status(client):
    """Test health endpoint returns unhealthy status when all models fail."""
    from src.api import health
    from src.services.model_manager import ModelManager

    # Mock all models as failed
    health.MODELS_REGISTRY = {
        "yolov8n.pt": {"status": "failed", "type": "yolo", "error": "Download failed"},
        "large-v3": {"status": "failed", "type": "whisper", "error": "Download failed"},
    }
    health.MODEL_MANAGER = ModelManager()

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "unhealthy"
    assert len(data["models_loaded"]) == 0


def test_health_gpu_info_included(client):
    """Test health endpoint includes GPU info."""
    from src.api import health
    from src.services.model_manager import ModelManager

    health.MODELS_REGISTRY = {
        "yolov8n.pt": {"status": "ready", "type": "yolo"},
    }
    health.MODEL_MANAGER = ModelManager()

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["gpu_available"], bool)
    # GPU info may be None if no GPU available
    if data["gpu_available"]:
        assert data["gpu_device_name"] is not None
        assert data["gpu_memory_total_mb"] is not None
        assert data["gpu_memory_used_mb"] is not None
