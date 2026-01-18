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


class Transcription(Base):
    __tablename__ = "transcriptions"

    segment_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    start = Column(Float, nullable=False)  # Start time in seconds
    end = Column(Float, nullable=False)  # End time in seconds
    confidence = Column(Float)  # Transcription confidence score
    speaker = Column(String)  # Speaker ID for multi-speaker audio
    created_at = Column(DateTime, server_default=func.now())


class Scene(Base):
    __tablename__ = "scenes"

    scene_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    scene = Column(Integer, nullable=False)  # Scene number/index
    start = Column(Float, nullable=False)  # Start time in seconds
    end = Column(Float, nullable=False)  # End time in seconds
    thumbnail_path = Column(String)  # Path to scene thumbnail image
    created_at = Column(DateTime, server_default=func.now())


class Object(Base):
    __tablename__ = "objects"

    object_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    label = Column(String, nullable=False, index=True)  # Object class label
    timestamps = Column(JSON, nullable=False)  # Timestamps where object appears
    bounding_boxes = Column(JSON, nullable=False)  # Bounding box coordinates
    created_at = Column(DateTime, server_default=func.now())


class Face(Base):
    __tablename__ = "faces"

    face_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    person_id = Column(String)  # Person identifier for clustering
    timestamps = Column(JSON, nullable=False)  # Timestamps where face appears
    bounding_boxes = Column(JSON, nullable=False)  # Bounding box coordinates
    confidence = Column(Float)  # Face detection confidence
    created_at = Column(DateTime, server_default=func.now())


class Topic(Base):
    __tablename__ = "topics"

    topic_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    label = Column(String, nullable=False, index=True)  # Topic label
    keywords = Column(JSON, nullable=False)  # Related keywords
    relevance_score = Column(Float, nullable=False)  # Topic relevance score
    timestamps = Column(JSON, nullable=False)  # Timestamps where topic appears
    created_at = Column(DateTime, server_default=func.now())


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
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error = Column(Text)  # Error message if failed
