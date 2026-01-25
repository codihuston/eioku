"""API controller for path configuration and video discovery."""

# Set up logging - use root logger
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..repositories.path_config_repository import SQLAlchemyPathConfigRepository
from ..repositories.video_repository import SqlVideoRepository
from ..services.job_producer import JobProducer
from ..services.path_config_manager import PathConfigManager
from ..services.processing_profiles import ProfileManager
from ..services.video_discovery_service import VideoDiscoveryService

logger = logging.getLogger()

router = APIRouter(tags=["paths"])

# Create a single ProfileManager instance that persists
_profile_manager = None


def get_path_config_manager(db: Session = Depends(get_db)) -> PathConfigManager:
    """Get PathConfigManager instance."""
    path_repo = SQLAlchemyPathConfigRepository(db)
    return PathConfigManager(path_repo)


def get_video_discovery_service(db: Session = Depends(get_db)) -> VideoDiscoveryService:
    """Get VideoDiscoveryService instance with JobProducer for task enqueueing."""
    path_repo = SQLAlchemyPathConfigRepository(db)
    video_repo = SqlVideoRepository(db)
    path_manager = PathConfigManager(path_repo)
    job_producer = JobProducer()
    return VideoDiscoveryService(path_manager, video_repo, job_producer)


def get_profile_manager() -> ProfileManager:
    """Get ProfileManager singleton instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager


@router.get("/paths")
async def list_paths(manager: PathConfigManager = Depends(get_path_config_manager)):
    """List all configured paths."""
    paths = manager.list_paths()
    return [
        {
            "path_id": p.path_id,
            "path": p.path,
            "recursive": p.recursive,
            "added_at": p.added_at.isoformat(),
        }
        for p in paths
    ]


@router.post("/paths")
async def add_path(
    path_data: dict, manager: PathConfigManager = Depends(get_path_config_manager)
):
    """Add a new path configuration."""
    try:
        path_config = manager.add_path(
            path_data.get("path"), path_data.get("recursive", True)
        )
        return {
            "path_id": path_config.path_id,
            "path": path_config.path,
            "recursive": path_config.recursive,
            "added_at": path_config.added_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/paths/discover")
async def discover_videos(
    discovery_service: VideoDiscoveryService = Depends(get_video_discovery_service),
):
    """Discover videos in all configured paths and auto-create ML tasks."""
    # Force logger level to INFO for proper logging
    logger.setLevel(logging.INFO)

    logger.info("Discovery endpoint called")
    try:
        logger.info("Getting configured paths...")
        discovered_videos = discovery_service.discover_videos()
        logger.info(f"Discovery completed. Found {len(discovered_videos)} videos")

        # Initialize JobProducer for task enqueueing
        if not discovery_service.job_producer:
            raise RuntimeError("JobProducer not initialized in VideoDiscoveryService")

        await discovery_service.job_producer.initialize()

        # Auto-create and enqueue ML tasks for each discovered video
        tasks_created = 0
        for video in discovered_videos:
            if video.status == "discovered":
                try:
                    await discovery_service.discover_and_queue_tasks(video.file_path)
                    tasks_created += 6  # 6 tasks per video
                    logger.info(
                        f"Auto-created and queued 6 ML tasks for video {video.video_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to auto-create tasks for video {video.video_id}: {e}",
                        exc_info=True,
                    )
                    # Continue with next video instead of failing entire discovery

        await discovery_service.job_producer.close()

        result = {
            "message": f"Discovered {len(discovered_videos)} videos",
            "tasks_created": tasks_created,
            "videos": [
                {
                    "video_id": v.video_id,
                    "filename": v.filename,
                    "file_path": v.file_path,
                    "file_size": v.file_size,
                    "status": v.status,
                }
                for v in discovered_videos
            ],
        }
        logger.info(f"Returning result with {len(result['videos'])} videos")
        return result
    except Exception as e:
        logger.error(f"Discovery failed with error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.post("/paths/validate")
async def validate_existing_videos(
    discovery_service: VideoDiscoveryService = Depends(get_video_discovery_service),
):
    """Validate that existing videos still exist on filesystem."""
    try:
        missing_videos = discovery_service.validate_existing_videos()
        return {
            "message": f"Found {len(missing_videos)} missing videos",
            "missing_videos": [
                {
                    "video_id": v.video_id,
                    "filename": v.filename,
                    "file_path": v.file_path,
                    "status": v.status,
                }
                for v in missing_videos
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/profiles")
async def list_processing_profiles(
    profile_manager: ProfileManager = Depends(get_profile_manager),
):
    """List all available processing profiles."""
    try:
        profiles = profile_manager.list_profiles()
        return {
            "profiles": [
                {"name": name, "description": description}
                for name, description in profiles.items()
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list profiles: {str(e)}"
        )


@router.get("/profiles/{profile_name}")
async def get_processing_profile(
    profile_name: str,
    profile_manager: ProfileManager = Depends(get_profile_manager),
):
    """Get detailed configuration for a specific processing profile."""
    try:
        profile = profile_manager.get_profile(profile_name)
        return profile.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")


@router.post("/profiles/load")
async def load_custom_profile(
    file_data: dict,
    profile_manager: ProfileManager = Depends(get_profile_manager),
):
    """Load a custom profile from provided configuration data."""
    try:
        file_path = file_data.get("file_path")
        if not file_path:
            raise ValueError("file_path is required")

        profile = profile_manager.load_profile(file_path)
        return {
            "message": f"Successfully loaded profile '{profile.name}'",
            "profile": profile.to_dict(),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load profile: {str(e)}")
