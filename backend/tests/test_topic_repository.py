"""Test TopicRepository implementation."""

import tempfile
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Video
from src.domain.models import Topic
from src.repositories.topic_repository import SQLAlchemyTopicRepository


def test_topic_repository_crud():
    """Test Topic repository CRUD operations."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)
    session = session_local()

    try:
        # Create test videos first
        video1 = Video(
            video_id="video_1",
            file_path="/test/video1.mp4",
            filename="video1.mp4",
            last_modified=datetime.utcnow(),
            status="completed",
        )
        video2 = Video(
            video_id="video_2",
            file_path="/test/video2.mp4",
            filename="video2.mp4",
            last_modified=datetime.utcnow(),
            status="completed",
        )
        session.add_all([video1, video2])
        session.commit()

        repo = SQLAlchemyTopicRepository(session)

        # Create test topic
        topic = Topic(
            topic_id="topic_1",
            video_id="video_1",
            label="technology",
            keywords=["AI", "machine learning", "automation"],
            relevance_score=0.85,
            timestamps=[10.5, 45.2, 120.8],
        )

        # Test save
        saved_topic = repo.save(topic)
        assert saved_topic.topic_id == "topic_1"
        assert saved_topic.video_id == "video_1"
        assert saved_topic.label == "technology"
        assert len(saved_topic.keywords) == 3
        assert saved_topic.relevance_score == 0.85
        assert len(saved_topic.timestamps) == 3
        assert saved_topic.created_at is not None

        # Test find_by_video_id
        topics = repo.find_by_video_id("video_1")
        assert len(topics) == 1
        assert topics[0].topic_id == "topic_1"

        # Test find_by_label
        tech_topics = repo.find_by_label("video_1", "technology")
        assert len(tech_topics) == 1
        assert tech_topics[0].label == "technology"

        # Test find_by_label with different label
        other_topics = repo.find_by_label("video_1", "sports")
        assert len(other_topics) == 0

        # Add more topics for testing
        topic2 = Topic(
            topic_id="topic_2",
            video_id="video_1",
            label="business",
            keywords=["startup", "investment", "growth"],
            relevance_score=0.72,
            timestamps=[25.1, 80.5],
        )

        topic3 = Topic(
            topic_id="topic_3",
            video_id="video_2",
            label="technology",
            keywords=["software", "development", "coding"],
            relevance_score=0.91,
            timestamps=[15.3, 60.7, 95.2, 140.1],
        )

        repo.save(topic2)
        repo.save(topic3)

        # Test multiple topics for same video (should be ordered by relevance)
        video1_topics = repo.find_by_video_id("video_1")
        assert len(video1_topics) == 2
        assert video1_topics[0].relevance_score >= video1_topics[1].relevance_score

        # Test aggregated topics
        aggregated = repo.get_aggregated_topics()
        assert len(aggregated) == 2  # technology and business

        # Technology should have higher average relevance
        tech_agg = next(t for t in aggregated if t["label"] == "technology")
        business_agg = next(t for t in aggregated if t["label"] == "business")

        assert tech_agg["video_count"] == 2
        assert business_agg["video_count"] == 1
        assert tech_agg["avg_relevance"] > business_agg["avg_relevance"]
        assert tech_agg["max_relevance"] == 0.91

        # Test delete_by_video_id
        deleted = repo.delete_by_video_id("video_1")
        assert deleted is True

        # Verify deletion
        topics_after_delete = repo.find_by_video_id("video_1")
        assert len(topics_after_delete) == 0

        # Video 2 topics should still exist
        video2_topics = repo.find_by_video_id("video_2")
        assert len(video2_topics) == 1

        # Test delete non-existent video
        deleted_none = repo.delete_by_video_id("nonexistent")
        assert deleted_none is False

    finally:
        session.close()


def test_topic_domain_methods():
    """Test Topic domain model methods."""
    topic = Topic(
        topic_id="topic_1",
        video_id="video_1",
        label="science",
        keywords=["physics", "chemistry", "biology", "research"],
        relevance_score=0.88,
        timestamps=[5.2, 30.7, 65.1, 95.4, 120.8],
    )

    # Test occurrence count
    assert topic.get_occurrence_count() == 5

    # Test first appearance
    assert topic.get_first_appearance() == 5.2

    # Test last appearance
    assert topic.get_last_appearance() == 120.8

    # Test high relevance
    assert topic.is_highly_relevant() is True
    assert topic.is_highly_relevant(0.9) is False

    # Test low relevance topic
    low_relevance_topic = Topic(
        topic_id="topic_2",
        video_id="video_1",
        label="misc",
        keywords=["random"],
        relevance_score=0.45,
        timestamps=[10.0],
    )

    assert low_relevance_topic.is_highly_relevant() is False
    assert low_relevance_topic.get_occurrence_count() == 1
    assert low_relevance_topic.get_first_appearance() == 10.0
    assert low_relevance_topic.get_last_appearance() == 10.0

    # Test empty timestamps
    empty_topic = Topic(
        topic_id="topic_3",
        video_id="video_1",
        label="empty",
        keywords=[],
        relevance_score=0.0,
        timestamps=[],
    )

    assert empty_topic.get_occurrence_count() == 0
    assert empty_topic.get_first_appearance() is None
    assert empty_topic.get_last_appearance() is None
