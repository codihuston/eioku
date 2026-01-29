#!/usr/bin/env python3
"""Show all tasks with their status, video info, and Redis queue status."""

import os
import sys
from datetime import datetime

import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/eioku"
)
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Redis connection - use valkey service name in docker-compose
REDIS_URL = os.getenv("REDIS_URL", "redis://valkey:6379")
redis_client = redis.from_url(REDIS_URL)


def get_queued_task_ids() -> set:
    """Get all task IDs currently queued in Redis."""
    queued = set()
    try:
        # Get all keys from Redis
        for key in redis_client.scan_iter("arq:*"):
            # arq:queue contains job IDs
            if b"queue" in key:
                jobs = redis_client.lrange(key, 0, -1)
                for job_id in jobs:
                    queued.add(job_id.decode() if isinstance(job_id, bytes) else job_id)
    except Exception as e:
        print(f"Warning: Could not query Redis: {e}", file=sys.stderr)
    return queued


def format_timestamp(ts: datetime | None) -> str:
    """Format timestamp for display."""
    if ts is None:
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def get_viewer_url(video_id: str, task_type: str) -> str | None:
    """Generate viewer URL for completed tasks."""
    base_url = "http://localhost:8080"

    if task_type == "scene_detection":
        return f"{base_url}/?view=scenes&video={video_id}"
    elif task_type == "face_detection":
        return f"{base_url}/?view=faces&video={video_id}"

    return None


def get_api_urls(task_id: str, video_id: str, task_type: str) -> dict:
    """Generate API URLs for task and artifacts."""
    base_url = "http://localhost:8080/api/v1"

    # Map task types to artifact type filters
    artifact_type_map = {
        "scene_detection": "scene",
        "face_detection": "face.detection",
        "object_detection": "object.detection",
        "ocr": "ocr.text",
        "place_detection": "place.classification",
        "transcription": "transcript.segment",
    }

    artifact_type = artifact_type_map.get(task_type, task_type)

    urls = {
        "video": f"{base_url}/videos/{video_id}",
        "artifacts": f"{base_url}/videos/{video_id}/artifacts?type={artifact_type}",
    }

    return urls


def main():
    """Show all tasks with their status and queue info."""
    session = Session()
    queued_ids = get_queued_task_ids()

    try:
        # Query all tasks with their video info
        query = text(
            """
            SELECT
                t.task_id,
                t.task_type,
                t.status,
                t.priority,
                t.created_at,
                t.started_at,
                t.completed_at,
                t.error,
                v.video_id,
                v.file_path,
                v.filename,
                v.status as video_status
            FROM tasks t
            JOIN videos v ON t.video_id = v.video_id
            ORDER BY v.video_id, t.created_at DESC
        """
        )

        result = session.execute(query)
        rows = result.fetchall()

        if not rows:
            print("No tasks found.")
            return

        # Print header
        print("\n" + "=" * 220)
        print(
            f"{'Task ID':<36} | {'Type':<15} | {'Status':<12} | {'Queued':<7} | "
            f"{'Video ID':<36} | {'Filename':<30} | {'Created':<19} | "
            f"{'Viewer URL':<50}"
        )
        print("=" * 220)

        # Print rows
        for row in rows:
            task_id = row[0]
            task_type = row[1]
            status = row[2]
            video_id = row[8]
            filename = row[10]

            # Check if queued
            is_queued = task_id in queued_ids
            queued_str = "✓" if is_queued else "✗"

            # Truncate filename
            filename_str = (
                (filename[:27] + "...")
                if filename and len(filename) > 30
                else filename or ""
            )

            # Generate viewer URL if task is completed
            viewer_url = ""
            if status == "completed":
                url = get_viewer_url(video_id, task_type)
                if url:
                    viewer_url = url[:47] + "..." if len(url) > 50 else url

            created_at = row[4]
            print(
                f"{task_id:<36} | {task_type:<15} | {status:<12} | "
                f"{queued_str:<7} | {video_id:<36} | {filename_str:<30} | "
                f"{format_timestamp(created_at):<19} | {viewer_url:<50}"
            )

        print("=" * 220)
        print(f"\nTotal tasks: {len(rows)}")
        print(f"Queued: {sum(1 for row in rows if row[0] in queued_ids)}")
        print(f"Pending: {sum(1 for row in rows if row[2] == 'pending')}")
        print(f"Running: {sum(1 for row in rows if row[2] == 'running')}")
        print(f"Completed: {sum(1 for row in rows if row[2] == 'completed')}")
        print(f"Failed: {sum(1 for row in rows if row[2] == 'failed')}")

        # Group tasks by video
        videos_map: dict = {}
        for row in rows:
            video_id = row[8]
            task_type = row[1]
            filename = row[10]

            if video_id not in videos_map:
                videos_map[video_id] = {
                    "filename": filename,
                    "task_types": set(),
                }
            videos_map[video_id]["task_types"].add(task_type)

        # Print videos with their artifact URLs
        print("\n" + "=" * 220)
        print("Videos and Artifact URLs:")
        print("=" * 220)
        for video_id, video_info in sorted(videos_map.items()):
            filename = video_info["filename"]
            print(f"\nVideo: {filename}")
            print(f"  ID: {video_id}")
            print(f"  Video URL: http://localhost:8080/api/v1/videos/{video_id}")

            # Show artifact URLs for each task type
            for task_type in sorted(video_info["task_types"]):
                apis = get_api_urls("", video_id, task_type)
                print(f"  {task_type:<20} | {apis['artifacts']}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
