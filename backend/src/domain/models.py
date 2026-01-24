from datetime import datetime


class Video:
    """Domain model for Video - pure business object."""

    def __init__(
        self,
        video_id: str,
        file_path: str,
        filename: str,
        last_modified: datetime,
        status: str = "pending",
        file_hash: str | None = None,
        duration: float | None = None,
        file_size: int | None = None,
        processed_at: datetime | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        self.video_id = video_id
        self.file_path = file_path
        self.filename = filename
        self.last_modified = last_modified
        self.status = status
        self.file_hash = file_hash
        self.duration = duration
        self.file_size = file_size
        self.processed_at = processed_at
        self.created_at = created_at
        self.updated_at = updated_at

    def is_processed(self) -> bool:
        """Check if video has been processed."""
        return self.status == "completed"

    def mark_as_processing(self) -> None:
        """Mark video as currently being processed."""
        self.status = "processing"

    def mark_as_completed(self, processed_at: datetime) -> None:
        """Mark video as processing completed."""
        self.status = "completed"
        self.processed_at = processed_at

    def mark_as_failed(self) -> None:
        """Mark video processing as failed."""
        self.status = "failed"


class PathConfig:
    """Domain model for PathConfig - pure business object."""

    def __init__(
        self,
        path_id: str,
        path: str,
        recursive: bool = True,
        added_at: datetime | None = None,
    ):
        self.path_id = path_id
        self.path = path
        self.recursive = recursive
        self.added_at = added_at or datetime.utcnow()

    def is_recursive(self) -> bool:
        """Check if path should be scanned recursively."""
        return self.recursive


class Task:
    """Domain model for Task - pure business object."""

    def __init__(
        self,
        task_id: str,
        video_id: str,
        task_type: str,
        status: str = "pending",
        priority: int = 1,
        dependencies: list[str] | None = None,
        created_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
    ):
        self.task_id = task_id
        self.video_id = video_id
        self.task_type = task_type
        self.status = status
        self.priority = priority
        self.dependencies = dependencies or []
        self.created_at = created_at or datetime.utcnow()
        self.started_at = started_at
        self.completed_at = completed_at
        self.error = error

    def is_pending(self) -> bool:
        """Check if task is pending."""
        return self.status == "pending"

    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == "running"

    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == "completed"

    def is_failed(self) -> bool:
        """Check if task has failed."""
        return self.status == "failed"

    def start(self) -> None:
        """Mark task as started."""
        self.status = "running"
        self.started_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark task as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark task as failed with error message."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error = error
