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
