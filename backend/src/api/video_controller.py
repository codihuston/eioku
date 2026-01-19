from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..api.schemas import VideoCreateSchema, VideoResponseSchema, VideoUpdateSchema
from ..database.connection import get_db
from ..domain.models import Video
from ..repositories.video_repository import SqlVideoRepository
from ..services.video_service import VideoService

router = APIRouter(prefix="/videos", tags=["videos"])


def get_video_service(session: Session = Depends(get_db)) -> VideoService:
    """Dependency injection for VideoService."""
    repository = SqlVideoRepository(session)
    return VideoService(repository)


@router.post(
    "/", response_model=VideoResponseSchema, status_code=status.HTTP_201_CREATED
)
async def create_video(
    video_data: VideoCreateSchema, service: VideoService = Depends(get_video_service)
) -> VideoResponseSchema:
    """Create a new video for processing."""
    try:
        # Convert schema to domain model
        domain_video = Video(
            video_id=video_data.video_id,
            file_path=video_data.file_path,
            filename=video_data.filename,
            file_hash=video_data.file_hash,
            last_modified=video_data.last_modified,
            duration=video_data.duration,
            file_size=video_data.file_size,
        )

        created_video = service.create_video(domain_video)
        return VideoResponseSchema.model_validate(created_video.__dict__)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/{video_id}", response_model=VideoResponseSchema)
async def get_video(
    video_id: str, service: VideoService = Depends(get_video_service)
) -> VideoResponseSchema:
    """Get video by ID."""
    video = service.get_video(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video not found"
        )

    return VideoResponseSchema.model_validate(video.__dict__)


@router.get("/", response_model=list[VideoResponseSchema])
async def list_videos(
    status: str = None, service: VideoService = Depends(get_video_service)
) -> list[VideoResponseSchema]:
    """List videos, optionally filtered by status."""
    if status:
        videos = service.get_videos_by_status(status)
    else:
        # Return all videos
        videos = service.get_all_videos()

    return [VideoResponseSchema.model_validate(video.__dict__) for video in videos]


@router.patch("/{video_id}", response_model=VideoResponseSchema)
async def update_video(
    video_id: str,
    update_data: VideoUpdateSchema,
    service: VideoService = Depends(get_video_service),
) -> VideoResponseSchema:
    """Update video metadata."""
    if update_data.status:
        video = service.update_video_status(video_id, update_data.status)
    else:
        video = service.get_video(video_id)

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video not found"
        )

    return VideoResponseSchema.model_validate(video.__dict__)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: str, service: VideoService = Depends(get_video_service)
) -> None:
    """Delete video."""
    success = service.delete_video(video_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video not found"
        )


@router.get("/{video_id}/transcription")
async def get_video_transcription(video_id: str, session: Session = Depends(get_db)):
    """Get transcription for a video."""
    from ..repositories.transcription_repository import SqlTranscriptionRepository

    transcription_repo = SqlTranscriptionRepository(session)
    segments = transcription_repo.find_by_video_id(video_id)

    if not segments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found for this video",
        )

    # Combine segments into full text
    full_text = " ".join(seg.text for seg in segments)

    # Convert segments to dict format
    segments_data = [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "confidence": seg.confidence if hasattr(seg, "confidence") else None,
        }
        for seg in segments
    ]

    return {
        "video_id": video_id,
        "full_text": full_text,
        "segments": segments_data,
        "segment_count": len(segments),
        "created_at": segments[0].created_at if segments else None,
    }


@router.get("/{video_id}/scenes")
async def get_video_scenes(video_id: str, session: Session = Depends(get_db)):
    """Get detected scenes for a video."""
    from ..repositories.scene_repository import SqlSceneRepository

    scene_repo = SqlSceneRepository(session)
    scenes = scene_repo.find_by_video_id(video_id)

    if not scenes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenes not found for this video",
        )

    # Convert scenes to dict format
    scenes_data = [
        {
            "scene_id": scene.scene_id,
            "scene": scene.scene,
            "start": scene.start,
            "end": scene.end,
            "duration": scene.get_duration(),
            "thumbnail_path": scene.thumbnail_path,
            "created_at": scene.created_at,
        }
        for scene in scenes
    ]

    # Calculate statistics
    durations = [scene.get_duration() for scene in scenes]
    total_duration = scenes[-1].end if scenes else 0.0

    return {
        "video_id": video_id,
        "scenes": scenes_data,
        "scene_count": len(scenes),
        "total_duration": total_duration,
        "avg_scene_length": sum(durations) / len(durations) if durations else 0.0,
        "min_scene_length": min(durations) if durations else 0.0,
        "max_scene_length": max(durations) if durations else 0.0,
        "created_at": scenes[0].created_at if scenes else None,
    }


@router.get("/{video_id}/objects")
async def get_video_objects(
    video_id: str, label: str = None, session: Session = Depends(get_db)
):
    """Get detected objects for a video, optionally filtered by label."""
    from ..repositories.object_repository import SqlObjectRepository

    object_repo = SqlObjectRepository(session)

    # Filter by label if provided
    if label:
        objects = object_repo.find_by_label(video_id, label)
    else:
        objects = object_repo.find_by_video_id(video_id)

    if not objects:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Objects not found for this video",
        )

    # Convert objects to dict format
    objects_data = [
        {
            "object_id": obj.object_id,
            "label": obj.label,
            "occurrences": obj.get_occurrence_count(),
            "first_appearance": obj.get_first_appearance(),
            "last_appearance": obj.get_last_appearance(),
            "timestamps": obj.timestamps,
            "bounding_boxes": obj.bounding_boxes,
            "created_at": obj.created_at,
        }
        for obj in objects
    ]

    # Calculate statistics
    total_occurrences = sum(obj.get_occurrence_count() for obj in objects)
    unique_labels = len(set(obj.label for obj in objects))

    return {
        "video_id": video_id,
        "objects": objects_data,
        "unique_labels": unique_labels,
        "total_occurrences": total_occurrences,
        "created_at": objects[0].created_at if objects else None,
    }
