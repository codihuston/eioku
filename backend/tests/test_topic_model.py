import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Topic, Video


def test_topic_model_creation():
    """Test that Topic model can be created with JSON fields."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        Base.metadata.create_all(engine)

        session_class = sessionmaker(bind=engine)
        session = session_class()

        # Create a video first
        video = Video(
            video_id="test-video-1",
            file_path="/path/to/video.mp4",
            filename="video.mp4",
            last_modified=datetime.now(),
            status="pending"
        )
        session.add(video)
        session.commit()

        # Create topic extraction result
        topic = Topic(
            topic_id="topic-1",
            video_id="test-video-1",
            label="technology",
            keywords=["AI", "machine learning", "software", "development"],
            relevance_score=0.85,
            timestamps=[10.5, 45.2, 120.8, 180.3]
        )

        session.add(topic)
        session.commit()

        # Query it back
        retrieved = session.query(Topic).filter_by(topic_id="topic-1").first()
        assert retrieved is not None
        assert retrieved.label == "technology"
        assert retrieved.relevance_score == 0.85
        assert "AI" in retrieved.keywords
        assert len(retrieved.timestamps) == 4

        session.close()
