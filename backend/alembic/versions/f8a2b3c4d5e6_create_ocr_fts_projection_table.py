"""create_ocr_fts_projection_table

Revision ID: f8a2b3c4d5e6
Revises: 325d54cd2340
Create Date: 2026-01-20 23:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f8a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "325d54cd2340"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Determine if we're using PostgreSQL or SQLite
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    if is_postgresql:
        # PostgreSQL: Create table with tsvector column
        op.create_table(
            "ocr_fts",
            sa.Column("artifact_id", sa.String(), nullable=False),
            sa.Column("asset_id", sa.String(), nullable=False),
            sa.Column("start_ms", sa.Integer(), nullable=False),
            sa.Column("end_ms", sa.Integer(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column(
                "text_tsv",
                postgresql.TSVECTOR(),
                sa.Computed("to_tsvector('english', text)", persisted=True),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.artifact_id"]),
        )

        # GIN index for fast full-text search
        op.create_index(
            "idx_ocr_fts_tsv",
            "ocr_fts",
            ["text_tsv"],
            postgresql_using="gin",
        )

        # Index for asset filtering
        op.create_index("idx_ocr_fts_asset", "ocr_fts", ["asset_id", "start_ms"])
    else:
        # SQLite: Create FTS5 virtual table
        # Note: FTS5 tables are created differently in SQLite
        op.execute(
            """
            CREATE VIRTUAL TABLE ocr_fts USING fts5(
                artifact_id UNINDEXED,
                asset_id UNINDEXED,
                start_ms UNINDEXED,
                end_ms UNINDEXED,
                text,
                content='',
                tokenize='porter unicode61'
            )
            """
        )

        # Create a regular table to store the metadata for SQLite
        op.create_table(
            "ocr_fts_metadata",
            sa.Column("artifact_id", sa.String(), nullable=False),
            sa.Column("asset_id", sa.String(), nullable=False),
            sa.Column("start_ms", sa.Integer(), nullable=False),
            sa.Column("end_ms", sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.artifact_id"]),
            sa.PrimaryKeyConstraint("artifact_id"),
        )

        # Index for asset filtering in SQLite
        op.create_index(
            "idx_ocr_fts_metadata_asset",
            "ocr_fts_metadata",
            ["asset_id", "start_ms"],
        )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    if is_postgresql:
        # Drop indexes first
        op.drop_index("idx_ocr_fts_asset", "ocr_fts")
        op.drop_index("idx_ocr_fts_tsv", "ocr_fts")

        # Drop table
        op.drop_table("ocr_fts")
    else:
        # SQLite: Drop FTS5 virtual table and metadata table
        op.drop_index("idx_ocr_fts_metadata_asset", "ocr_fts_metadata")
        op.drop_table("ocr_fts_metadata")
        op.execute("DROP TABLE IF EXISTS ocr_fts")
