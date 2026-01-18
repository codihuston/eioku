"""Test FaceRepository implementation."""

import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Video
from src.domain.models import Face
from src.repositories.face_repository import SQLAlchemyFaceRepository


def test_face_repository_crud():
    """Test Face repository CRUD operations."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)
    session = session_local()

    try:
        # Create test video first
        video = Video(
            video_id="video_1",
            file_path="/test/video.mp4",
            filename="video.mp4",
            last_modified=datetime.utcnow(),
            status="completed",
        )
        session.add(video)
        session.commit()

        repo = SQLAlchemyFaceRepository(session)

        # Create test face
        face = Face(
            face_id="face_1",
            video_id="video_1",
            person_id="person_1",
            timestamps=[10.5, 25.3, 45.7],
            bounding_boxes=[
                {"x": 100, "y": 150, "width": 80, "height": 100},
                {"x": 120, "y": 160, "width": 85, "height": 105},
                {"x": 110, "y": 155, "width": 82, "height": 102},
            ],
            confidence=0.95,
        )

        # Test save
        saved_face = repo.save(face)
        assert saved_face.face_id == "face_1"
        assert saved_face.video_id == "video_1"
        assert saved_face.person_id == "person_1"
        assert len(saved_face.timestamps) == 3
        assert saved_face.confidence == 0.95
        assert saved_face.created_at is not None

        # Test find_by_video_id
        faces = repo.find_by_video_id("video_1")
        assert len(faces) == 1
        assert faces[0].face_id == "face_1"

        # Test find_by_person_id
        person_faces = repo.find_by_person_id("video_1", "person_1")
        assert len(person_faces) == 1
        assert person_faces[0].person_id == "person_1"

        # Test find_by_person_id with different person
        other_faces = repo.find_by_person_id("video_1", "person_2")
        assert len(other_faces) == 0

        # Add another face for same video, different person
        face2 = Face(
            face_id="face_2",
            video_id="video_1",
            person_id="person_2",
            timestamps=[15.2, 30.8],
            bounding_boxes=[
                {"x": 200, "y": 250, "width": 75, "height": 95},
                {"x": 210, "y": 260, "width": 78, "height": 98},
            ],
            confidence=0.88,
        )
        repo.save(face2)

        # Test multiple faces for same video
        all_faces = repo.find_by_video_id("video_1")
        assert len(all_faces) == 2

        # Test person-specific queries
        person1_faces = repo.find_by_person_id("video_1", "person_1")
        assert len(person1_faces) == 1
        assert person1_faces[0].face_id == "face_1"

        person2_faces = repo.find_by_person_id("video_1", "person_2")
        assert len(person2_faces) == 1
        assert person2_faces[0].face_id == "face_2"

        # Test delete_by_video_id
        deleted = repo.delete_by_video_id("video_1")
        assert deleted is True

        # Verify deletion
        faces_after_delete = repo.find_by_video_id("video_1")
        assert len(faces_after_delete) == 0

        # Test delete non-existent video
        deleted_none = repo.delete_by_video_id("nonexistent")
        assert deleted_none is False

    finally:
        session.close()


def test_face_domain_methods():
    """Test Face domain model methods."""
    face = Face(
        face_id="face_1",
        video_id="video_1",
        person_id="person_1",
        timestamps=[10.5, 25.3, 45.7, 60.1],
        bounding_boxes=[
            {"x": 100, "y": 150, "width": 80, "height": 100},
            {"x": 120, "y": 160, "width": 85, "height": 105},
            {"x": 110, "y": 155, "width": 82, "height": 102},
            {"x": 115, "y": 158, "width": 83, "height": 103},
        ],
        confidence=0.92,
    )

    # Test occurrence count
    assert face.get_occurrence_count() == 4

    # Test first appearance
    assert face.get_first_appearance() == 10.5

    # Test last appearance
    assert face.get_last_appearance() == 60.1

    # Test is_identified
    assert face.is_identified() is True

    # Test unidentified face
    unidentified_face = Face(
        face_id="face_2",
        video_id="video_1",
        person_id=None,
        timestamps=[20.0],
        bounding_boxes=[{"x": 50, "y": 75, "width": 60, "height": 80}],
        confidence=0.75,
    )

    assert unidentified_face.is_identified() is False
    assert unidentified_face.get_occurrence_count() == 1
    assert unidentified_face.get_first_appearance() == 20.0
    assert unidentified_face.get_last_appearance() == 20.0

    # Test empty timestamps
    empty_face = Face(
        face_id="face_3",
        video_id="video_1",
        person_id="person_3",
        timestamps=[],
        bounding_boxes=[],
        confidence=0.0,
    )

    assert empty_face.get_occurrence_count() == 0
    assert empty_face.get_first_appearance() is None
    assert empty_face.get_last_appearance() is None
