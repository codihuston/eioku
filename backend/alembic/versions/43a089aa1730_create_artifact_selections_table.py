"""create_artifact_selections_table

Revision ID: 43a089aa1730
Revises: 7d93a6c78758
Create Date: 2026-01-20 21:29:42.966339

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "43a089aa1730"
down_revision: str | Sequence[str] | None = "7d93a6c78758"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create artifact_selections table for selection policies
    op.create_table(
        "artifact_selections",
        sa.Column("asset_id", sa.String(), nullable=False),
        sa.Column("artifact_type", sa.String(), nullable=False),
        sa.Column("selection_mode", sa.String(), nullable=False),
        sa.Column("preferred_profile", sa.String(), nullable=True),
        sa.Column("pinned_run_id", sa.String(), nullable=True),
        sa.Column("pinned_artifact_id", sa.String(), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["videos.video_id"]),
        sa.PrimaryKeyConstraint("asset_id", "artifact_type"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop table
    op.drop_table("artifact_selections")
