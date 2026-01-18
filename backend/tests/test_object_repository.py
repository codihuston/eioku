import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Video
from src.domain.models import Object as ObjectDomain
from src.repositories.object_repository import SqlObjectRepository


def test_object_repository_crud():
    """Test ObjectRepository CRUD operations."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        Base.metadata.create_all(engine)

        session_class = sessionmaker(bind=engine)
        session = session_class()

        # Create video first
        video = Video(
            video_id="test-video-1",
            file_path="/test/video.mp4",
            filename="video.mp4",
            last_modified=datetime.now(),
            status="pending",
            file_hash=None,
        )
        session.add(video)
        session.commit()

        # Test repository
        repo = SqlObjectRepository(session)

        # Create objects
        person_obj = ObjectDomain(
            object_id="obj-1",
            video_id="test-video-1",
            label="person",
            timestamps=[1.5, 3.2, 5.8],
            bounding_boxes=[
                {"x": 100, "y": 50, "width": 200, "height": 300},
                {"x": 110, "y": 55, "width": 190, "height": 295},
                {"x": 105, "y": 52, "width": 195, "height": 298},
            ],
        )

        car_obj = ObjectDomain(
            object_id="obj-2",
            video_id="test-video-1",
            label="car",
            timestamps=[10.0, 15.5],
            bounding_boxes=[
                {"x": 50, "y": 200, "width": 300, "height": 150},
                {"x": 55, "y": 205, "width": 295, "height": 145},
            ],
        )

        # Save objects
        saved_person = repo.save(person_obj)
        saved_car = repo.save(car_obj)

        assert saved_person.get_occurrence_count() == 3
        assert saved_person.get_first_appearance() == 1.5
        assert saved_person.get_last_appearance() == 5.8
        assert saved_car.get_occurrence_count() == 2

        # Find by video ID
        objects = repo.find_by_video_id("test-video-1")
        assert len(objects) == 2
        labels = [obj.label for obj in objects]
        assert "person" in labels
        assert "car" in labels

        # Find by label
        persons = repo.find_by_label("test-video-1", "person")
        assert len(persons) == 1
        assert persons[0].label == "person"

        cars = repo.find_by_label("test-video-1", "car")
        assert len(cars) == 1

        dogs = repo.find_by_label("test-video-1", "dog")
        assert len(dogs) == 0

        # Delete by video ID
        deleted = repo.delete_by_video_id("test-video-1")
        assert deleted is True

        remaining = repo.find_by_video_id("test-video-1")
        assert len(remaining) == 0

        session.close()
