import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.main import app


@pytest.fixture(scope="module")
def test_db():
    """Create a temporary database for testing."""
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    # Create engine and tables
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    # Create session factory
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield testing_session_local

    # Cleanup
    os.unlink(db_path)


@pytest.fixture(scope="module")
def client(test_db):
    """Create test client with database dependency override."""

    def override_get_db():
        db = test_db()
        try:
            yield db
        finally:
            db.close()

    from src.database.connection import get_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_openapi_spec(client):
    """Test that OpenAPI spec is generated correctly."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    spec = response.json()
    assert spec["info"]["title"] == "Eioku - Semantic Video Search API"
    assert spec["info"]["version"] == "1.0.0"
    assert "paths" in spec
    assert "/v1/videos/" in spec["paths"]


def test_swagger_docs(client):
    """Test that Swagger UI is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_redoc_docs(client):
    """Test that ReDoc is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower()


def test_create_video(client):
    """Test creating a video via API."""
    video_data = {
        "video_id": "test-video-1",
        "file_path": "/test/video.mp4",
        "filename": "video.mp4",
        "last_modified": "2024-01-01T12:00:00",
        "duration": 120.5,
        "file_size": 1024000,
        "file_hash": "abc123def456",
    }

    response = client.post("/v1/videos/", json=video_data)
    assert response.status_code == 201

    data = response.json()
    assert data["video_id"] == "test-video-1"
    assert data["status"] == "pending"
    assert data["duration"] == 120.5


def test_create_video_duplicate(client):
    """Test creating duplicate video returns conflict."""
    video_data = {
        "video_id": "test-video-2",
        "file_path": "/test/duplicate.mp4",
        "filename": "duplicate.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }

    # Create first video
    response1 = client.post("/v1/videos/", json=video_data)
    assert response1.status_code == 201

    # Try to create duplicate
    response2 = client.post("/v1/videos/", json=video_data)
    assert response2.status_code == 409
    assert "already exists" in response2.json()["detail"]


def test_get_video(client):
    """Test getting a video by ID."""
    # Create video first
    video_data = {
        "video_id": "test-video-3",
        "file_path": "/test/get.mp4",
        "filename": "get.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # Get video
    response = client.get("/v1/videos/test-video-3")
    assert response.status_code == 200

    data = response.json()
    assert data["video_id"] == "test-video-3"
    assert data["filename"] == "get.mp4"


def test_get_video_not_found(client):
    """Test getting non-existent video returns 404."""
    response = client.get("/v1/videos/non-existent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_list_videos(client):
    """Test listing videos."""
    # Create a video first
    video_data = {
        "video_id": "test-video-4",
        "file_path": "/test/list.mp4",
        "filename": "list.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # List videos
    response = client.get("/v1/videos/")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    # Should have at least the video we just created
    video_ids = [v["video_id"] for v in data]
    assert "test-video-4" in video_ids


def test_update_video(client):
    """Test updating video status."""
    # Create video first
    video_data = {
        "video_id": "test-video-5",
        "file_path": "/test/update.mp4",
        "filename": "update.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # Update status
    update_data = {"status": "completed"}
    response = client.patch("/v1/videos/test-video-5", json=update_data)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "completed"
    assert data["processed_at"] is not None


def test_delete_video(client):
    """Test deleting a video."""
    # Create video first
    video_data = {
        "video_id": "test-video-6",
        "file_path": "/test/delete.mp4",
        "filename": "delete.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # Delete video
    response = client.delete("/v1/videos/test-video-6")
    assert response.status_code == 204

    # Verify it's gone
    get_response = client.get("/v1/videos/test-video-6")
    assert get_response.status_code == 404


def test_invalid_json_validation(client):
    """Test that invalid JSON is rejected with validation errors."""
    invalid_data = {
        "video_id": "",  # Empty string should fail validation
        "file_path": "/test/invalid.mp4",
        # Missing required fields
    }

    response = client.post("/v1/videos/", json=invalid_data)
    assert response.status_code == 422

    error_data = response.json()
    assert "detail" in error_data
    assert isinstance(error_data["detail"], list)


def test_get_video_scenes(client):
    """Test getting scenes for a video."""
    from src.database.models import Scene

    # Create video first
    video_data = {
        "video_id": "test-video-scenes",
        "file_path": "/test/scenes.mp4",
        "filename": "scenes.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # Add some scenes directly to the database
    from src.database.connection import get_db

    db = next(app.dependency_overrides[get_db]())

    scenes = [
        Scene(
            scene_id="scene_1",
            video_id="test-video-scenes",
            scene=1,
            start=0.0,
            end=5.5,
            thumbnail_path="/thumbnails/scene_1.jpg",
        ),
        Scene(
            scene_id="scene_2",
            video_id="test-video-scenes",
            scene=2,
            start=5.5,
            end=12.3,
            thumbnail_path="/thumbnails/scene_2.jpg",
        ),
    ]

    for scene in scenes:
        db.add(scene)
    db.commit()
    db.close()

    # Get scenes
    response = client.get("/v1/videos/test-video-scenes/scenes")
    assert response.status_code == 200

    data = response.json()
    assert data["video_id"] == "test-video-scenes"
    assert data["scene_count"] == 2
    assert len(data["scenes"]) == 2
    assert data["scenes"][0]["scene"] == 1
    assert data["scenes"][0]["start"] == 0.0
    assert data["scenes"][0]["end"] == 5.5
    assert data["scenes"][0]["duration"] == 5.5
    assert data["total_duration"] == 12.3
    assert data["avg_scene_length"] > 0
    assert data["min_scene_length"] > 0
    assert data["max_scene_length"] > 0


def test_get_video_scenes_not_found(client):
    """Test getting scenes for video with no scenes returns 404."""
    # Create video without scenes
    video_data = {
        "video_id": "test-video-no-scenes",
        "file_path": "/test/no-scenes.mp4",
        "filename": "no-scenes.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # Try to get scenes
    response = client.get("/v1/videos/test-video-no-scenes/scenes")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_video_objects(client):
    """Test getting detected objects for a video."""
    from src.database.models import Object as DbObject

    # Create video first
    video_data = {
        "video_id": "test-video-objects",
        "file_path": "/test/objects.mp4",
        "filename": "objects.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # Add some objects directly to the database
    from src.database.connection import get_db

    db = next(app.dependency_overrides[get_db]())

    objects = [
        DbObject(
            object_id="obj_1",
            video_id="test-video-objects",
            label="person",
            timestamps=[1.0, 2.0, 3.0],
            bounding_boxes=[
                {
                    "frame": 30,
                    "timestamp": 1.0,
                    "bbox": [10, 20, 30, 40],
                    "confidence": 0.95,
                },
                {
                    "frame": 60,
                    "timestamp": 2.0,
                    "bbox": [15, 25, 35, 45],
                    "confidence": 0.92,
                },
                {
                    "frame": 90,
                    "timestamp": 3.0,
                    "bbox": [12, 22, 32, 42],
                    "confidence": 0.88,
                },
            ],
        ),
        DbObject(
            object_id="obj_2",
            video_id="test-video-objects",
            label="car",
            timestamps=[1.5, 4.0],
            bounding_boxes=[
                {
                    "frame": 45,
                    "timestamp": 1.5,
                    "bbox": [100, 200, 300, 400],
                    "confidence": 0.97,
                },
                {
                    "frame": 120,
                    "timestamp": 4.0,
                    "bbox": [110, 210, 310, 410],
                    "confidence": 0.93,
                },
            ],
        ),
    ]

    for obj in objects:
        db.add(obj)
    db.commit()
    db.close()

    # Get objects
    response = client.get("/v1/videos/test-video-objects/objects")
    assert response.status_code == 200

    data = response.json()
    assert data["video_id"] == "test-video-objects"
    assert data["unique_labels"] == 2
    assert data["total_occurrences"] == 5  # 3 person + 2 car
    assert len(data["objects"]) == 2

    # Check person object
    person_obj = next(obj for obj in data["objects"] if obj["label"] == "person")
    assert person_obj["occurrences"] == 3
    assert person_obj["first_appearance"] == 1.0
    assert person_obj["last_appearance"] == 3.0
    assert len(person_obj["timestamps"]) == 3
    assert len(person_obj["bounding_boxes"]) == 3

    # Check car object
    car_obj = next(obj for obj in data["objects"] if obj["label"] == "car")
    assert car_obj["occurrences"] == 2
    assert car_obj["first_appearance"] == 1.5
    assert car_obj["last_appearance"] == 4.0


def test_get_video_objects_filtered_by_label(client):
    """Test getting detected objects filtered by label."""
    # Use the same video from previous test
    response = client.get("/v1/videos/test-video-objects/objects?label=person")
    assert response.status_code == 200

    data = response.json()
    assert data["video_id"] == "test-video-objects"
    assert data["unique_labels"] == 1
    assert data["total_occurrences"] == 3
    assert len(data["objects"]) == 1
    assert data["objects"][0]["label"] == "person"


def test_get_video_objects_not_found(client):
    """Test getting objects for video with no objects returns 404."""
    # Create video without objects
    video_data = {
        "video_id": "test-video-no-objects",
        "file_path": "/test/no-objects.mp4",
        "filename": "no-objects.mp4",
        "last_modified": "2024-01-01T12:00:00",
    }
    client.post("/v1/videos/", json=video_data)

    # Try to get objects
    response = client.get("/v1/videos/test-video-no-objects/objects")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_video_objects_label_not_found(client):
    """Test getting objects with non-existent label returns 404."""
    # Try to get objects with label that doesn't exist
    response = client.get("/v1/videos/test-video-objects/objects?label=airplane")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
