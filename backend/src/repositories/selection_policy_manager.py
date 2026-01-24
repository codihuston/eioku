"""Selection policy manager for artifact version selection."""

from datetime import datetime

from sqlalchemy.orm import Session

from ..database.models import ArtifactSelection
from ..domain.artifacts import SelectionPolicy


class SelectionPolicyManager:
    """Manages selection policies for artifacts."""

    def __init__(self, session: Session):
        """Initialize manager with database session."""
        self.session = session

    def get_policy(self, asset_id: str, artifact_type: str) -> SelectionPolicy | None:
        """
        Get selection policy for asset and artifact type.

        Args:
            asset_id: The asset (video) ID
            artifact_type: The artifact type (e.g., "transcript.segment")

        Returns:
            SelectionPolicy if found, None otherwise
        """
        entity = (
            self.session.query(ArtifactSelection)
            .filter(
                ArtifactSelection.asset_id == asset_id,
                ArtifactSelection.artifact_type == artifact_type,
            )
            .first()
        )

        if not entity:
            return None

        return self._to_domain(entity)

    def set_policy(self, policy: SelectionPolicy) -> SelectionPolicy:
        """
        Set or update selection policy.

        Args:
            policy: The selection policy to set

        Returns:
            The created or updated selection policy
        """
        existing = (
            self.session.query(ArtifactSelection)
            .filter(
                ArtifactSelection.asset_id == policy.asset_id,
                ArtifactSelection.artifact_type == policy.artifact_type,
            )
            .first()
        )

        if existing:
            # Update existing policy
            existing.selection_mode = policy.mode
            existing.preferred_profile = policy.preferred_profile
            existing.pinned_run_id = policy.pinned_run_id
            existing.pinned_artifact_id = policy.pinned_artifact_id
            existing.updated_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(existing)
            return self._to_domain(existing)
        else:
            # Create new policy
            entity = self._to_entity(policy)
            self.session.add(entity)
            self.session.commit()
            self.session.refresh(entity)
            return self._to_domain(entity)

    def get_default_policy(
        self, asset_id: str = "", artifact_type: str = ""
    ) -> SelectionPolicy | None:
        """
        Get default selection policy (latest).

        Args:
            asset_id: Optional asset ID for the default policy
            artifact_type: Optional artifact type for the default policy

        Returns:
            A default SelectionPolicy with mode="latest" if asset_id and artifact_type
            are provided, None otherwise (indicating no filtering should be applied)

        Note:
            Returns None when asset_id or artifact_type are empty strings to indicate
            that no default policy should be applied. This allows the query to return
            all artifacts without filtering by run_id.
        """
        # Return None if asset_id or artifact_type are empty
        # This indicates no default policy should be applied
        if not asset_id or not artifact_type:
            return None

        # Create a policy with mode="latest" for the given asset/type
        policy = object.__new__(SelectionPolicy)
        policy.asset_id = asset_id
        policy.artifact_type = artifact_type
        policy.mode = "latest"
        policy.preferred_profile = None
        policy.pinned_run_id = None
        policy.pinned_artifact_id = None
        policy.updated_at = datetime.utcnow()
        return policy

    def _to_domain(self, entity: ArtifactSelection) -> SelectionPolicy:
        """Convert database entity to domain model."""
        return SelectionPolicy(
            asset_id=entity.asset_id,
            artifact_type=entity.artifact_type,
            mode=entity.selection_mode,
            preferred_profile=entity.preferred_profile,
            pinned_run_id=entity.pinned_run_id,
            pinned_artifact_id=entity.pinned_artifact_id,
            updated_at=entity.updated_at,
        )

    def _to_entity(self, policy: SelectionPolicy) -> ArtifactSelection:
        """Convert domain model to database entity."""
        return ArtifactSelection(
            asset_id=policy.asset_id,
            artifact_type=policy.artifact_type,
            selection_mode=policy.mode,
            preferred_profile=policy.preferred_profile,
            pinned_run_id=policy.pinned_run_id,
            pinned_artifact_id=policy.pinned_artifact_id,
            updated_at=policy.updated_at or datetime.utcnow(),
        )
