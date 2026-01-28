"""Jump navigation service for deterministic artifact navigation."""

import json
import logging

from ..domain.artifacts import SelectionPolicy
from ..repositories.artifact_repository import ArtifactRepository
from ..repositories.selection_policy_manager import SelectionPolicyManager

logger = logging.getLogger(__name__)


class JumpNavigationService:
    """Service for jump navigation across artifacts."""

    def __init__(
        self,
        artifact_repo: ArtifactRepository,
        policy_manager: SelectionPolicyManager,
    ):
        """
        Initialize jump navigation service.

        Args:
            artifact_repo: Repository for artifact storage and retrieval
            policy_manager: Manager for selection policies
        """
        self.artifact_repo = artifact_repo
        self.policy_manager = policy_manager

    def jump_next(
        self,
        asset_id: str,
        artifact_type: str,
        from_ms: int,
        label: str | None = None,
        cluster_id: str | None = None,
        min_confidence: float = -float("inf"),
        selection: SelectionPolicy | None = None,
    ) -> dict | None:
        """
        Jump to next artifact occurrence.

        Args:
            asset_id: The asset (video) ID
            artifact_type: Type of artifact to jump to
            from_ms: Starting timestamp in milliseconds
            label: Optional label filter (for objects, places)
            cluster_id: Optional cluster ID filter (for faces)
            min_confidence: Minimum confidence threshold
            selection: Optional selection policy override

        Returns:
            Dictionary with jump_to (start_ms, end_ms) and artifact_ids,
            or None if no match found
        """
        logger.debug(
            f"jump_next: asset_id={asset_id}, type={artifact_type}, "
            f"from_ms={from_ms}, label={label}, cluster_id={cluster_id}"
        )

        # Get selection policy
        policy = (
            selection
            or self.policy_manager.get_policy(asset_id, artifact_type)
            or self.policy_manager.get_default_policy(asset_id, artifact_type)
        )

        # Get artifacts starting AT or AFTER the given timestamp
        # We need to query all artifacts and filter manually since the repo
        # uses >= instead of > for start_ms
        artifacts = self.artifact_repo.get_by_asset(
            asset_id=asset_id,
            artifact_type=artifact_type,
            selection=policy,
        )

        # Filter to only artifacts that start at or after from_ms
        artifacts = [a for a in artifacts if a.span_start_ms >= from_ms]

        # Filter by label/cluster if specified
        filtered = self._filter_artifacts(artifacts, label, cluster_id, min_confidence)

        if not filtered:
            logger.debug("No matching artifacts found for jump_next")
            return None

        # Return first match (earliest start time)
        artifact = filtered[0]
        logger.info(
            f"Jump next found artifact {artifact.artifact_id} at "
            f"{artifact.span_start_ms}ms"
        )

        return {
            "jump_to": {
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms,
            },
            "artifact_ids": [artifact.artifact_id],
        }

    def jump_prev(
        self,
        asset_id: str,
        artifact_type: str,
        from_ms: int,
        label: str | None = None,
        cluster_id: str | None = None,
        min_confidence: float = -float("inf"),
        selection: SelectionPolicy | None = None,
    ) -> dict | None:
        """
        Jump to previous artifact occurrence.

        Args:
            asset_id: The asset (video) ID
            artifact_type: Type of artifact to jump to
            from_ms: Starting timestamp in milliseconds
            label: Optional label filter (for objects, places)
            cluster_id: Optional cluster ID filter (for faces)
            min_confidence: Minimum confidence threshold
            selection: Optional selection policy override

        Returns:
            Dictionary with jump_to (start_ms, end_ms) and artifact_ids,
            or None if no match found
        """
        logger.debug(
            f"jump_prev: asset_id={asset_id}, type={artifact_type}, "
            f"from_ms={from_ms}, label={label}, cluster_id={cluster_id}"
        )

        # Get selection policy
        policy = (
            selection
            or self.policy_manager.get_policy(asset_id, artifact_type)
            or self.policy_manager.get_default_policy(asset_id, artifact_type)
        )

        # Get artifacts ending before the given timestamp
        # We need to query all artifacts and filter manually since the repo
        # uses <= instead of < for end_ms
        artifacts = self.artifact_repo.get_by_asset(
            asset_id=asset_id,
            artifact_type=artifact_type,
            selection=policy,
        )

        # Filter to only artifacts that end strictly before from_ms
        # Using < instead of <= prevents returning the same artifact when
        # jumping from a position within or at the end of that artifact
        artifacts = [a for a in artifacts if a.span_end_ms < from_ms]
        
        logger.debug(
            f"After filtering by span_end_ms < {from_ms}: "
            f"found {len(artifacts)} artifacts"
        )
        for a in artifacts:
            logger.debug(
                f"  - artifact {a.artifact_id}: {a.span_start_ms}-{a.span_end_ms}ms"
            )

        # Filter by label/cluster if specified
        filtered = self._filter_artifacts(artifacts, label, cluster_id, min_confidence)

        if not filtered:
            logger.debug("No matching artifacts found for jump_prev")
            return None

        # Return last match (latest end time before from_ms)
        # Artifacts are ordered by span_start_ms, so we take the last one
        artifact = filtered[-1]
        logger.info(
            f"Jump prev found artifact {artifact.artifact_id} at "
            f"{artifact.span_start_ms}ms"
        )

        return {
            "jump_to": {
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms,
            },
            "artifact_ids": [artifact.artifact_id],
        }

    def _filter_artifacts(self, artifacts, label, cluster_id, min_confidence):
        """
        Filter artifacts by label, cluster, and confidence.

        Args:
            artifacts: List of ArtifactEnvelope objects
            label: Optional label to filter by
            cluster_id: Optional cluster ID to filter by
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of artifacts
        """
        filtered = []

        for artifact in artifacts:
            try:
                payload = json.loads(artifact.payload_json)

                # Check confidence (skip if confidence is None)
                confidence = payload.get("confidence")
                if confidence is not None and confidence < min_confidence:
                    logger.debug(
                        f"Artifact {artifact.artifact_id} filtered out: "
                        f"confidence {confidence} < {min_confidence}"
                    )
                    continue

                # Check label (for objects, places)
                if label and payload.get("label") != label:
                    logger.debug(
                        f"Artifact {artifact.artifact_id} filtered out: "
                        f"label {payload.get('label')} != {label}"
                    )
                    continue

                # Check cluster (for faces)
                if cluster_id and payload.get("cluster_id") != cluster_id:
                    logger.debug(
                        f"Artifact {artifact.artifact_id} filtered out: "
                        f"cluster_id {payload.get('cluster_id')} != {cluster_id}"
                    )
                    continue

                filtered.append(artifact)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Failed to parse payload for artifact {artifact.artifact_id}: {e}"
                )
                continue

        logger.debug(f"Filtered {len(filtered)} artifacts from {len(artifacts)} total")
        return filtered
