"""create_places_table

Revision ID: 2c0b0659a6b8
Revises: 4022c4cef20d
Create Date: 2026-01-19 18:49:15.788766

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2c0b0659a6b8"
down_revision: str | Sequence[str] | None = "4022c4cef20d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "places",
        sa.Column("place_id", sa.String(), nullable=False),
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("timestamps", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("alternative_labels", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.video_id"]),
        sa.PrimaryKeyConstraint("place_id"),
    )
    op.create_index("ix_places_video_id", "places", ["video_id"])
    op.create_index("ix_places_label", "places", ["label"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_places_label", "places")
    op.drop_index("ix_places_video_id", "places")
    op.drop_table("places")
