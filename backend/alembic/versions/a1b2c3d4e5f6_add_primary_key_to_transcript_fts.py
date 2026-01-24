"""add_primary_key_to_transcript_fts

Revision ID: a1b2c3d4e5f6
Revises: f8a2b3c4d5e6
Create Date: 2026-01-24 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "f8a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Determine if we're using PostgreSQL or SQLite
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    if is_postgresql:
        # PostgreSQL: Add primary key constraint to artifact_id if it doesn't exist
        # Use raw SQL to check and add if needed
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints 
                    WHERE table_name = 'transcript_fts' 
                    AND constraint_name = 'pk_transcript_fts_artifact_id'
                ) THEN
                    ALTER TABLE transcript_fts
                    ADD CONSTRAINT pk_transcript_fts_artifact_id PRIMARY KEY (artifact_id);
                END IF;
            END $$;
            """
        )
    else:
        # SQLite: FTS5 tables don't support traditional primary keys
        # The metadata table already has the primary key
        pass


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    if is_postgresql:
        # Drop the primary key constraint
        op.execute(
            """
            ALTER TABLE transcript_fts
            DROP CONSTRAINT pk_transcript_fts_artifact_id
            """
        )
    else:
        # SQLite: No changes needed
        pass
