"""SQLAlchemy implementation of TopicRepository."""

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database.models import Topic as TopicEntity
from ..domain.models import Topic
from .interfaces import TopicRepository


class SQLAlchemyTopicRepository(TopicRepository):
    """SQLAlchemy implementation of TopicRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, topic: Topic) -> Topic:
        """Save topic to database."""
        entity = TopicEntity(
            topic_id=topic.topic_id,
            video_id=topic.video_id,
            label=topic.label,
            keywords=topic.keywords,
            relevance_score=topic.relevance_score,
            timestamps=topic.timestamps,
            created_at=topic.created_at or datetime.utcnow(),
        )

        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)

        return self._entity_to_domain(entity)

    def find_by_video_id(self, video_id: str) -> list[Topic]:
        """Find all topics for a video."""
        entities = (
            self.session.query(TopicEntity)
            .filter(TopicEntity.video_id == video_id)
            .order_by(TopicEntity.relevance_score.desc())
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def find_by_label(self, video_id: str, label: str) -> list[Topic]:
        """Find topics by label within a video."""
        entities = (
            self.session.query(TopicEntity)
            .filter(TopicEntity.video_id == video_id, TopicEntity.label == label)
            .order_by(TopicEntity.relevance_score.desc())
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def get_aggregated_topics(self) -> list[dict]:
        """Get aggregated topics across all videos."""
        results = (
            self.session.query(
                TopicEntity.label,
                func.count(TopicEntity.topic_id).label("video_count"),
                func.avg(TopicEntity.relevance_score).label("avg_relevance"),
                func.max(TopicEntity.relevance_score).label("max_relevance"),
            )
            .group_by(TopicEntity.label)
            .order_by(func.avg(TopicEntity.relevance_score).desc())
            .all()
        )

        return [
            {
                "label": result.label,
                "video_count": result.video_count,
                "avg_relevance": float(result.avg_relevance),
                "max_relevance": float(result.max_relevance),
            }
            for result in results
        ]

    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all topics for a video."""
        deleted_count = (
            self.session.query(TopicEntity)
            .filter(TopicEntity.video_id == video_id)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _entity_to_domain(self, entity: TopicEntity) -> Topic:
        """Convert database entity to domain model."""
        return Topic(
            topic_id=entity.topic_id,
            video_id=entity.video_id,
            label=entity.label,
            keywords=entity.keywords,
            relevance_score=entity.relevance_score,
            timestamps=entity.timestamps,
            created_at=entity.created_at,
        )
