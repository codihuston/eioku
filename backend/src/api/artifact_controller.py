"""API controller for artifact-based endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..api.schemas import (
    ArtifactResponseSchema,
    FindResponseSchema,
    JumpResponseSchema,
    ProfileInfoSchema,
    ProfilesResponseSchema,
)
from ..database.connection import get_db
from ..domain.artifacts import SelectionPolicy
from ..domain.schema_registry import SchemaRegistry
from ..repositories.artifact_repository import SqlArtifactRepository
from ..repositories.selection_policy_manager import SelectionPolicyManager
from ..services.find_within_video_service import FindWithinVideoService
from ..services.jump_navigation_service import JumpNavigationService
from ..services.projection_sync_service import ProjectionSyncService

router = APIRouter(prefix="/videos", tags=["artifacts"])


def get_jump_navigation_service(
    session: Session = Depends(get_db),
) -> JumpNavigationService:
    """Dependency injection for JumpNavigationService."""
    schema_registry = SchemaRegistry()
    projection_sync = ProjectionSyncService(session)
    artifact_repo = SqlArtifactRepository(session, schema_registry, projection_sync)
    policy_manager = SelectionPolicyManager(session)
    return JumpNavigationService(artifact_repo, policy_manager)


def get_find_within_video_service(
    session: Session = Depends(get_db),
) -> FindWithinVideoService:
    """Dependency injection for FindWithinVideoService."""
    policy_manager = SelectionPolicyManager(session)
    return FindWithinVideoService(session, policy_manager)


def get_artifact_repository(
    session: Session = Depends(get_db),
) -> SqlArtifactRepository:
    """Dependency injection for ArtifactRepository."""
    schema_registry = SchemaRegistry()
    projection_sync = ProjectionSyncService(session)
    return SqlArtifactRepository(session, schema_registry, projection_sync)


def get_selection_policy_manager(
    session: Session = Depends(get_db),
) -> SelectionPolicyManager:
    """Dependency injection for SelectionPolicyManager."""
    return SelectionPolicyManager(session)


@router.get("/{video_id}/jump", response_model=JumpResponseSchema)
async def jump_to_artifact(
    video_id: str,
    kind: str = Query(
        ...,
        description=(
            "Artifact type to jump to: scene, transcript, object, face, place, ocr"
        ),
    ),
    direction: str = Query(..., description="Jump direction: next or prev"),
    from_ms: int = Query(..., description="Starting timestamp in milliseconds"),
    label: str | None = Query(None, description="Filter by label (for object, place)"),
    face_cluster_id: str | None = Query(
        None, description="Filter by face cluster ID (for face)"
    ),
    min_confidence: float = Query(
        0.0, description="Minimum confidence threshold", ge=0.0, le=1.0
    ),
    selection: str | None = Query(
        None,
        description=("Selection mode: default, pinned, latest, profile, best_quality"),
    ),
    profile: str | None = Query(
        None, description="Model profile: fast, balanced, high_quality"
    ),
    service: JumpNavigationService = Depends(get_jump_navigation_service),
) -> JumpResponseSchema:
    """
    Jump to next or previous artifact occurrence.

    Supports all artifact types with optional filtering by label, cluster,
    and confidence.
    """
    # Map kind to artifact_type
    artifact_type_map = {
        "scene": "scene",
        "transcript": "transcript.segment",
        "object": "object.detection",
        "face": "face.detection",
        "place": "place.classification",
        "ocr": "ocr.text",
    }

    if kind not in artifact_type_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid kind: {kind}. "
                f"Must be one of: {', '.join(artifact_type_map.keys())}"
            ),
        )

    artifact_type = artifact_type_map[kind]

    # Validate direction
    if direction not in ["next", "prev"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid direction: {direction}. Must be 'next' or 'prev'",
        )

    # Build selection policy if specified
    selection_policy = None
    if selection or profile:
        selection_policy = SelectionPolicy(
            asset_id=video_id,
            artifact_type=artifact_type,
            mode=selection or "default",
            preferred_profile=profile,
        )

    # Call appropriate service method
    if direction == "next":
        result = service.jump_next(
            asset_id=video_id,
            artifact_type=artifact_type,
            from_ms=from_ms,
            label=label,
            cluster_id=face_cluster_id,
            min_confidence=min_confidence,
            selection=selection_policy,
        )
    else:
        result = service.jump_prev(
            asset_id=video_id,
            artifact_type=artifact_type,
            from_ms=from_ms,
            label=label,
            cluster_id=face_cluster_id,
            min_confidence=min_confidence,
            selection=selection_policy,
        )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {kind} artifact found",
        )

    return JumpResponseSchema(**result)


@router.get("/{video_id}/find", response_model=FindResponseSchema)
async def find_within_video(
    video_id: str,
    q: str = Query(..., description="Search query text", alias="q"),
    direction: str = Query(..., description="Search direction: next or prev"),
    from_ms: int = Query(..., description="Starting timestamp in milliseconds"),
    source: str = Query(
        "all",
        description="Search source: transcript, ocr, or all",
    ),
    service: FindWithinVideoService = Depends(get_find_within_video_service),
) -> FindResponseSchema:
    """
    Find keyword occurrences within a video.

    Searches transcript and/or OCR text for the specified query.
    Returns matches with highlighted snippets.
    """
    # Validate direction
    if direction not in ["next", "prev"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid direction: {direction}. Must be 'next' or 'prev'",
        )

    # Validate source
    if source not in ["transcript", "ocr", "all"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid source: {source}. " f"Must be 'transcript', 'ocr', or 'all'"
            ),
        )

    # Call appropriate service method
    if direction == "next":
        matches = service.find_next(
            asset_id=video_id,
            query=q,
            from_ms=from_ms,
            source=source,
        )
    else:
        matches = service.find_prev(
            asset_id=video_id,
            query=q,
            from_ms=from_ms,
            source=source,
        )

    return FindResponseSchema(matches=matches)


@router.get("/{video_id}/artifacts", response_model=list[ArtifactResponseSchema])
async def get_artifacts(
    video_id: str,
    type: str | None = Query(None, description="Filter by artifact type", alias="type"),
    from_ms: int | None = Query(
        None, description="Filter by start time (milliseconds)"
    ),
    to_ms: int | None = Query(None, description="Filter by end time (milliseconds)"),
    selection: str | None = Query(
        None,
        description=("Selection mode: default, pinned, latest, profile, best_quality"),
    ),
    profile: str | None = Query(
        None, description="Model profile: fast, balanced, high_quality"
    ),
    artifact_repo: SqlArtifactRepository = Depends(get_artifact_repository),
    policy_manager: SelectionPolicyManager = Depends(get_selection_policy_manager),
) -> list[ArtifactResponseSchema]:
    """
    Get artifacts for a video with optional filtering.

    Supports filtering by type, time range, and selection policy.
    Returns artifacts with their full payloads.
    """
    # Build selection policy if specified
    selection_policy = None
    if selection or profile:
        selection_policy = SelectionPolicy(
            asset_id=video_id,
            artifact_type=type or "",
            mode=selection or "default",
            preferred_profile=profile,
        )
    elif type:
        # Get default policy for this artifact type
        selection_policy = policy_manager.get_policy(
            video_id, type
        ) or policy_manager.get_default_policy(video_id, type)

    # Query artifacts
    artifacts = artifact_repo.get_by_asset(
        asset_id=video_id,
        artifact_type=type,
        start_ms=from_ms,
        end_ms=to_ms,
        selection=selection_policy,
    )

    # Convert to response schema
    import json

    response = []
    for artifact in artifacts:
        payload = json.loads(artifact.payload_json)
        response.append(
            ArtifactResponseSchema(
                artifact_id=artifact.artifact_id,
                asset_id=artifact.asset_id,
                artifact_type=artifact.artifact_type,
                schema_version=artifact.schema_version,
                span_start_ms=artifact.span_start_ms,
                span_end_ms=artifact.span_end_ms,
                payload=payload,
                producer=artifact.producer,
                producer_version=artifact.producer_version,
                model_profile=artifact.model_profile,
                run_id=artifact.run_id,
                created_at=artifact.created_at,
            )
        )

    return response


@router.get("/{video_id}/profiles", response_model=ProfilesResponseSchema)
async def get_available_profiles(
    video_id: str,
    artifact_type: str = Query(..., description="Artifact type to query profiles for"),
    artifact_repo: SqlArtifactRepository = Depends(get_artifact_repository),
) -> ProfilesResponseSchema:
    """
    Get available model profiles for a video and artifact type.

    Returns information about which model profiles have been run for this
    video and artifact type, including artifact counts and run IDs.
    """
    # Query all artifacts for this video and type
    artifacts = artifact_repo.get_by_asset(
        asset_id=video_id,
        artifact_type=artifact_type,
    )

    # Group by model_profile
    profile_map: dict[str, dict] = {}
    for artifact in artifacts:
        profile = artifact.model_profile
        if profile not in profile_map:
            profile_map[profile] = {
                "artifact_count": 0,
                "run_ids": set(),
            }
        profile_map[profile]["artifact_count"] += 1
        profile_map[profile]["run_ids"].add(artifact.run_id)

    # Convert to response format
    profiles = [
        ProfileInfoSchema(
            profile=profile,
            artifact_count=data["artifact_count"],
            run_ids=sorted(list(data["run_ids"])),
        )
        for profile, data in sorted(profile_map.items())
    ]

    return ProfilesResponseSchema(
        video_id=video_id,
        artifact_type=artifact_type,
        profiles=profiles,
    )
