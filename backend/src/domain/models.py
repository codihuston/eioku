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


class Transcription:
    """Domain model for Transcription - pure business object."""

    def __init__(
        self,
        segment_id: str,
        video_id: str,
        text: str,
        start: float,
        end: float,
        confidence: float | None = None,
        speaker: str | None = None,
        created_at: datetime | None = None,
    ):
        self.segment_id = segment_id
        self.video_id = video_id
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
        self.speaker = speaker
        self.created_at = created_at

    def get_duration(self) -> float:
        """Get duration of transcription segment in seconds."""
        return self.end - self.start


class Scene:
    """Domain model for Scene - pure business object."""

    def __init__(
        self,
        scene_id: str,
        video_id: str,
        scene: int,
        start: float,
        end: float,
        thumbnail_path: str | None = None,
        created_at: datetime | None = None,
    ):
        self.scene_id = scene_id
        self.video_id = video_id
        self.scene = scene
        self.start = start
        self.end = end
        self.thumbnail_path = thumbnail_path
        self.created_at = created_at

    def get_duration(self) -> float:
        """Get duration of scene in seconds."""
        return self.end - self.start

    def has_thumbnail(self) -> bool:
        """Check if scene has a thumbnail."""
        return self.thumbnail_path is not None


class Object:
    """Domain model for Object - pure business object."""

    def __init__(
        self,
        object_id: str,
        video_id: str,
        label: str,
        timestamps: list[float],
        bounding_boxes: list[dict],
        created_at: datetime | None = None,
    ):
        self.object_id = object_id
        self.video_id = video_id
        self.label = label
        self.timestamps = timestamps
        self.bounding_boxes = bounding_boxes
        self.created_at = created_at

    def get_occurrence_count(self) -> int:
        """Get number of times object appears in video."""
        return len(self.timestamps)

    def get_first_appearance(self) -> float | None:
        """Get timestamp of first appearance."""
        return min(self.timestamps) if self.timestamps else None

    def get_last_appearance(self) -> float | None:
        """Get timestamp of last appearance."""
        return max(self.timestamps) if self.timestamps else None


class Face:
    """Domain model for Face - pure business object."""

    def __init__(
        self,
        face_id: str,
        video_id: str,
        person_id: str | None,
        timestamps: list[float],
        bounding_boxes: list[dict],
        confidence: float,
        created_at: datetime | None = None,
    ):
        self.face_id = face_id
        self.video_id = video_id
        self.person_id = person_id
        self.timestamps = timestamps
        self.bounding_boxes = bounding_boxes
        self.confidence = confidence
        self.created_at = created_at

    def get_occurrence_count(self) -> int:
        """Get number of times face appears in video."""
        return len(self.timestamps)

    def get_first_appearance(self) -> float | None:
        """Get timestamp of first appearance."""
        return min(self.timestamps) if self.timestamps else None

    def get_last_appearance(self) -> float | None:
        """Get timestamp of last appearance."""
        return max(self.timestamps) if self.timestamps else None

    def is_identified(self) -> bool:
        """Check if face has been identified with a person ID."""
        return self.person_id is not None


class Topic:
    """Domain model for Topic - pure business object."""

    def __init__(
        self,
        topic_id: str,
        video_id: str,
        label: str,
        keywords: list[str],
        relevance_score: float,
        timestamps: list[float],
        created_at: datetime | None = None,
    ):
        self.topic_id = topic_id
        self.video_id = video_id
        self.label = label
        self.keywords = keywords
        self.relevance_score = relevance_score
        self.timestamps = timestamps
        self.created_at = created_at

    def get_occurrence_count(self) -> int:
        """Get number of times topic appears in video."""
        return len(self.timestamps)

    def get_first_appearance(self) -> float | None:
        """Get timestamp of first appearance."""
        return min(self.timestamps) if self.timestamps else None

    def get_last_appearance(self) -> float | None:
        """Get timestamp of last appearance."""
        return max(self.timestamps) if self.timestamps else None

    def is_highly_relevant(self, threshold: float = 0.8) -> bool:
        """Check if topic has high relevance score."""
        return self.relevance_score >= threshold


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
