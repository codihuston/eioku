"""create_video_locations_projection_table

Revision ID: e7f8a9b0c1d2
Revises: d1e2f3a4b5c6
Create Date: 2026-01-28 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7f8a9b0c1d2"
down_revision: str | Sequence[str] | None = "d1e2f3a4b5c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create video_locations projection table
    op.create_table(
        "video_locations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("altitude", sa.Float(), nullable=True),
        sa.Column("country", sa.String(), nullable=True),
        sa.Column("state", sa.String(), nullable=True),
        sa.Column("city", sa.String(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.video_id"]),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.artifact_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_id"),
    )

    # Index on video_id for efficient video-based queries
    op.create_index("idx_video_locations_video_id", "video_locations", ["video_id"])

    # Index on latitude for geo-queries
    op.create_index("idx_video_locations_latitude", "video_locations", ["latitude"])

    # Index on longitude for geo-queries
    op.create_index(
        "idx_video_locations_longitude", "video_locations", ["longitude"]
    )

    # Indexes on location fields for searchability
    op.create_index("idx_video_locations_country", "video_locations", ["country"])
    op.create_index("idx_video_locations_state", "video_locations", ["state"])
    op.create_index("idx_video_locations_city", "video_locations", ["city"])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index("idx_video_locations_city", "video_locations")
    op.drop_index("idx_video_locations_state", "video_locations")
    op.drop_index("idx_video_locations_country", "video_locations")
    op.drop_index("idx_video_locations_longitude", "video_locations")
    op.drop_index("idx_video_locations_latitude", "video_locations")
    op.drop_index("idx_video_locations_asset_id", "video_locations")

    # Drop table
    op.drop_table("video_locations")
