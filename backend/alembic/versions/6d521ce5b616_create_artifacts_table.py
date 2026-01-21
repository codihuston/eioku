"""create_artifacts_table

Revision ID: 6d521ce5b616
Revises: 4022c4cef20d
Create Date: 2026-01-20 21:25:50.110743

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6d521ce5b616"
down_revision: str | Sequence[str] | None = "4022c4cef20d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Determine if we're using PostgreSQL or SQLite
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    # Use JSONB for PostgreSQL, JSON for SQLite
    json_type = postgresql.JSONB if is_postgresql else sa.JSON

    # Create artifacts table
    op.create_table(
        "artifacts",
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("asset_id", sa.String(), nullable=False),
        sa.Column("artifact_type", sa.String(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("span_start_ms", sa.Integer(), nullable=False),
        sa.Column("span_end_ms", sa.Integer(), nullable=False),
        sa.Column("payload_json", json_type, nullable=False),
        sa.Column("producer", sa.String(), nullable=False),
        sa.Column("producer_version", sa.String(), nullable=False),
        sa.Column("model_profile", sa.String(), nullable=False),
        sa.Column("config_hash", sa.String(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["videos.video_id"]),
        sa.PrimaryKeyConstraint("artifact_id"),
    )

    # Composite indexes for common query patterns
    op.create_index(
        "idx_artifacts_asset_type_start",
        "artifacts",
        ["asset_id", "artifact_type", "span_start_ms"],
    )

    op.create_index(
        "idx_artifacts_asset_type_profile_start",
        "artifacts",
        ["asset_id", "artifact_type", "model_profile", "span_start_ms"],
    )

    op.create_index(
        "idx_artifacts_type_created", "artifacts", ["artifact_type", "created_at"]
    )

    op.create_index("idx_artifacts_run", "artifacts", ["run_id"])

    # GIN index for JSONB payload (PostgreSQL only)
    if is_postgresql:
        op.create_index(
            "idx_artifacts_payload_gin",
            "artifacts",
            ["payload_json"],
            postgresql_using="gin",
        )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    if is_postgresql:
        op.drop_index("idx_artifacts_payload_gin", "artifacts")

    op.drop_index("idx_artifacts_run", "artifacts")
    op.drop_index("idx_artifacts_type_created", "artifacts")
    op.drop_index("idx_artifacts_asset_type_profile_start", "artifacts")
    op.drop_index("idx_artifacts_asset_type_start", "artifacts")

    # Drop table
    op.drop_table("artifacts")
