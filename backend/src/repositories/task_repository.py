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
            language=task.language,
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

    def find_by_id(self, task_id: str) -> Task | None:
        """Find task by ID."""
        entity = (
            self.session.query(TaskEntity).filter(TaskEntity.task_id == task_id).first()
        )
        return self._entity_to_domain(entity) if entity else None

    def find_all(self) -> list[Task]:
        """Find all tasks."""
        entities = (
            self.session.query(TaskEntity)
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
            language=entity.language,
            created_at=entity.created_at,
            started_at=entity.started_at,
            completed_at=entity.completed_at,
            error=entity.error,
        )

    def find_by_video_and_type(self, video_id: str, task_type: str) -> list[Task]:
        """Find tasks by video ID and task type."""
        entities = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.video_id == video_id)
            .filter(TaskEntity.task_type == task_type)
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def find_by_video_type_language(
        self, video_id: str, task_type: str, language: str | None
    ) -> Task | None:
        """Find a task by video ID, task type, and language.

        Args:
            video_id: Video ID to search for
            task_type: Task type to search for
            language: Language code (None for language-agnostic tasks)

        Returns:
            Task if found, None otherwise
        """
        query = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.video_id == video_id)
            .filter(TaskEntity.task_type == task_type)
        )

        if language is None:
            query = query.filter(TaskEntity.language.is_(None))
        else:
            query = query.filter(TaskEntity.language == language)

        entity = query.first()
        return self._entity_to_domain(entity) if entity else None

    def find_by_video_and_status(self, video_id: str, status: str) -> list[Task]:
        """Find tasks by video ID and status."""
        entities = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.video_id == video_id)
            .filter(TaskEntity.status == status)
            .all()
        )
        return [self._entity_to_domain(entity) for entity in entities]

    def update(self, task: Task) -> Task:
        """Update task in database."""
        entity = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.task_id == task.task_id)
            .first()
        )

        if not entity:
            raise ValueError(f"Task not found: {task.task_id}")

        entity.status = task.status
        entity.priority = task.priority
        entity.dependencies = task.dependencies
        entity.started_at = task.started_at
        entity.completed_at = task.completed_at
        entity.error = task.error

        self.session.commit()
        self.session.refresh(entity)

        return self._entity_to_domain(entity)

    def atomic_dequeue_pending_task(self, task_type: str) -> Task | None:
        """Atomically dequeue a pending task using SELECT FOR UPDATE.

        This ensures only one worker can claim a task at a time.
        """
        # Use SELECT FOR UPDATE SKIP LOCKED to atomically claim a task
        # SKIP LOCKED means if a row is locked, skip it and try the next one
        entity = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.task_type == task_type)
            .filter(TaskEntity.status == "pending")
            .order_by(TaskEntity.priority.desc(), TaskEntity.created_at.asc())
            .with_for_update(skip_locked=True)
            .first()
        )

        if not entity:
            return None

        # Mark as running immediately within the same transaction
        entity.status = "running"
        entity.started_at = datetime.utcnow()
        self.session.commit()

        return self._entity_to_domain(entity)
