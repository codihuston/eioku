"""create_runs_table

Revision ID: 7d93a6c78758
Revises: 6d521ce5b616
Create Date: 2026-01-20 21:29:11.303306

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7d93a6c78758"
down_revision: str | Sequence[str] | None = "6d521ce5b616"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create runs table for pipeline execution tracking
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("asset_id", sa.String(), nullable=False),
        sa.Column("pipeline_profile", sa.String(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["asset_id"], ["videos.video_id"]),
        sa.PrimaryKeyConstraint("run_id"),
    )

    # Indexes for common query patterns
    op.create_index("idx_runs_asset_status", "runs", ["asset_id", "status"])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index("idx_runs_asset_status", "runs")

    # Drop table
    op.drop_table("runs")
