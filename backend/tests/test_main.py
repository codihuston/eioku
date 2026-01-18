from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root():
    """Test hello world endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Eioku API is running"}


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
