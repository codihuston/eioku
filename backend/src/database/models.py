from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.sql import func

from .connection import Base


class Video(Base):
    __tablename__ = "videos"

    video_id = Column(String, primary_key=True)
    file_path = Column(String, nullable=False, unique=True, index=True)
    filename = Column(String, nullable=False)
    file_hash = Column(String, index=True)  # SHA-256 hash for deduplication
    duration = Column(Float)  # Duration in seconds
    file_size = Column(Integer)  # File size in bytes
    processed_at = Column(DateTime)
    last_modified = Column(DateTime, nullable=False)
    status = Column(String, nullable=False, default="pending", index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class PathConfig(Base):
    __tablename__ = "path_configs"

    path_id = Column(String, primary_key=True)
    path = Column(String, nullable=False, unique=True)  # File system path
    recursive = Column(String, nullable=False, default="true")  # Scan recursively
    added_at = Column(DateTime, server_default=func.now())


class Task(Base):
    __tablename__ = "tasks"

    task_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    task_type = Column(String, nullable=False, index=True)  # transcription, scene, etc
    status = Column(String, nullable=False, default="pending", index=True)  # Status
    priority = Column(Integer, nullable=False, default=5)  # 1=highest, 10=lowest
    dependencies = Column(JSON)  # List of task_ids that must complete first
    language = Column(String, nullable=True, index=True)  # ISO 639-1 language code
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error = Column(Text)  # Error message if failed


class Artifact(Base):
    """SQLAlchemy entity for artifact envelope storage."""

    __tablename__ = "artifacts"

    artifact_id = Column(String, primary_key=True)
    asset_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    artifact_type = Column(String, nullable=False, index=True)
    schema_version = Column(Integer, nullable=False)
    span_start_ms = Column(Integer, nullable=False)
    span_end_ms = Column(Integer, nullable=False)
    payload_json = Column(JSON, nullable=False)  # JSONB in PostgreSQL, JSON in SQLite
    producer = Column(String, nullable=False)
    producer_version = Column(String, nullable=False)
    model_profile = Column(String, nullable=False, index=True)
    config_hash = Column(String, nullable=False)
    input_hash = Column(String, nullable=False)
    run_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class Run(Base):
    """SQLAlchemy entity for pipeline run tracking."""

    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    asset_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    pipeline_profile = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime)
    status = Column(String, nullable=False, index=True)
    error = Column(Text)


class ArtifactSelection(Base):
    """SQLAlchemy entity for artifact selection policies."""

    __tablename__ = "artifact_selections"

    asset_id = Column(
        String, ForeignKey("videos.video_id"), nullable=False, primary_key=True
    )
    artifact_type = Column(String, nullable=False, primary_key=True)
    selection_mode = Column(String, nullable=False)
    preferred_profile = Column(String)
    pinned_run_id = Column(String)
    pinned_artifact_id = Column(String)
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SceneRange(Base):
    """SQLAlchemy entity for scene_ranges projection table."""

    __tablename__ = "scene_ranges"

    artifact_id = Column(
        String, ForeignKey("artifacts.artifact_id"), nullable=False, primary_key=True
    )
    asset_id = Column(String, nullable=False, index=True)
    scene_index = Column(Integer, nullable=False, index=True)
    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)


class ObjectLabel(Base):
    """SQLAlchemy entity for object_labels projection table."""

    __tablename__ = "object_labels"

    artifact_id = Column(
        String, ForeignKey("artifacts.artifact_id"), nullable=False, primary_key=True
    )
    asset_id = Column(String, nullable=False, index=True)
    label = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False, index=True)
    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)


class FaceCluster(Base):
    """SQLAlchemy entity for face_clusters projection table."""

    __tablename__ = "face_clusters"

    artifact_id = Column(
        String, ForeignKey("artifacts.artifact_id"), nullable=False, primary_key=True
    )
    asset_id = Column(String, nullable=False, index=True)
    cluster_id = Column(String, index=True)
    confidence = Column(Float, nullable=False, index=True)
    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)
