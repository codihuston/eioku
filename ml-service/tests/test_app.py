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
