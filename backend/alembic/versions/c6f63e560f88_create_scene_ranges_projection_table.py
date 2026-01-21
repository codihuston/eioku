"""create_scene_ranges_projection_table

Revision ID: c6f63e560f88
Revises: ed9b9d2014a1
Create Date: 2026-01-20 22:53:35.649171

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c6f63e560f88"
down_revision: str | Sequence[str] | None = "ed9b9d2014a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create scene_ranges projection table
    op.create_table(
        "scene_ranges",
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("asset_id", sa.String(), nullable=False),
        sa.Column("scene_index", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.artifact_id"]),
        sa.PrimaryKeyConstraint("artifact_id"),
    )

    # Index for asset and scene_index queries
    op.create_index(
        "idx_scene_ranges_asset_index", "scene_ranges", ["asset_id", "scene_index"]
    )

    # Index for time-based queries
    op.create_index(
        "idx_scene_ranges_asset_start", "scene_ranges", ["asset_id", "start_ms"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index("idx_scene_ranges_asset_start", "scene_ranges")
    op.drop_index("idx_scene_ranges_asset_index", "scene_ranges")

    # Drop table
    op.drop_table("scene_ranges")
