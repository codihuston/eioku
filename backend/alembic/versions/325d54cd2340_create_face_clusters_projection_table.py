"""create_face_clusters_projection_table

Revision ID: 325d54cd2340
Revises: 10a71bc9d989
Create Date: 2026-01-20 23:13:34.736107

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "325d54cd2340"
down_revision: str | Sequence[str] | None = "10a71bc9d989"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create face_clusters projection table
    op.create_table(
        "face_clusters",
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("asset_id", sa.String(), nullable=False),
        sa.Column("cluster_id", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.artifact_id"]),
        sa.PrimaryKeyConstraint("artifact_id"),
    )

    # Index for asset, cluster, and time-based queries
    op.create_index(
        "idx_face_clusters_asset_cluster_start",
        "face_clusters",
        ["asset_id", "cluster_id", "start_ms"],
    )

    # Index for confidence filtering
    op.create_index(
        "idx_face_clusters_confidence", "face_clusters", ["confidence"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index("idx_face_clusters_confidence", "face_clusters")
    op.drop_index("idx_face_clusters_asset_cluster_start", "face_clusters")

    # Drop table
    op.drop_table("face_clusters")
