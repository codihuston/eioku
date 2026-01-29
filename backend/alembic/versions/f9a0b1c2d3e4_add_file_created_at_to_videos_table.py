"""add_file_created_at_to_videos_table

Revision ID: f9a0b1c2d3e4
Revises: e7f8a9b0c1d2
Create Date: 2026-01-28 10:05:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f9a0b1c2d3e4"
down_revision: str | Sequence[str] | None = "e7f8a9b0c1d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add file_created_at column to videos table (nullable for existing videos)
    op.add_column(
        "videos",
        sa.Column("file_created_at", sa.DateTime(), nullable=True),
    )

    # Add index on file_created_at for efficient "next/prev video" queries
    op.create_index(
        "idx_videos_file_created_at", "videos", ["file_created_at"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop index first
    op.drop_index("idx_videos_file_created_at", "videos")

    # Drop column
    op.drop_column("videos", "file_created_at")
