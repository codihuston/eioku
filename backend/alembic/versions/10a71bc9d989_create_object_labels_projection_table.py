"""create_object_labels_projection_table

Revision ID: 10a71bc9d989
Revises: c6f63e560f88
Create Date: 2026-01-20 23:05:30.219253

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "10a71bc9d989"
down_revision: str | Sequence[str] | None = "c6f63e560f88"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create object_labels projection table
    op.create_table(
        "object_labels",
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("asset_id", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.artifact_id"]),
        sa.PrimaryKeyConstraint("artifact_id"),
    )

    # Index for asset, label, and time-based queries
    op.create_index(
        "idx_object_labels_asset_label_start",
        "object_labels",
        ["asset_id", "label", "start_ms"],
    )

    # Index for confidence filtering
    op.create_index(
        "idx_object_labels_confidence", "object_labels", ["confidence"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index("idx_object_labels_confidence", "object_labels")
    op.drop_index("idx_object_labels_asset_label_start", "object_labels")

    # Drop table
    op.drop_table("object_labels")
