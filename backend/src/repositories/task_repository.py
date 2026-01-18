"""SQLAlchemy implementation of TaskRepository."""

from datetime import datetime

from sqlalchemy.orm import Session

from ..database.models import Task as TaskEntity
from ..domain.models import Task
from .interfaces import TaskRepository


class SQLAlchemyTaskRepository(TaskRepository):
    """SQLAlchemy implementation of TaskRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, task: Task) -> Task:
        """Save task to database."""
        entity = TaskEntity(
            task_id=task.task_id,
            video_id=task.video_id,
            task_type=task.task_type,
            status=task.status,
            priority=task.priority,
            dependencies=task.dependencies,
            created_at=task.created_at or datetime.utcnow(),
            started_at=task.started_at,
            completed_at=task.completed_at,
            error=task.error,
        )

        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)

        return self._entity_to_domain(entity)

    def find_by_video_id(self, video_id: str) -> list[Task]:
        """Find all tasks for a video."""
        entities = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.video_id == video_id)
            .order_by(TaskEntity.priority.desc(), TaskEntity.created_at.asc())
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def find_by_status(self, status: str) -> list[Task]:
        """Find tasks by status."""
        entities = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.status == status)
            .order_by(TaskEntity.priority.desc(), TaskEntity.created_at.asc())
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def find_by_task_type(self, task_type: str) -> list[Task]:
        """Find tasks by type."""
        entities = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.task_type == task_type)
            .order_by(TaskEntity.priority.desc(), TaskEntity.created_at.asc())
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all tasks for a video."""
        deleted_count = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.video_id == video_id)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _entity_to_domain(self, entity: TaskEntity) -> Task:
        """Convert database entity to domain model."""
        return Task(
            task_id=entity.task_id,
            video_id=entity.video_id,
            task_type=entity.task_type,
            status=entity.status,
            priority=entity.priority,
            dependencies=entity.dependencies or [],
            created_at=entity.created_at,
            started_at=entity.started_at,
            completed_at=entity.completed_at,
            error=entity.error,
        )
