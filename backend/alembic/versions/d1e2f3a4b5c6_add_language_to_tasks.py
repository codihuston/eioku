"""Add language column to tasks table

Revision ID: d1e2f3a4b5c6
Revises: b2c3d4e5f6a7
Create Date: 2026-01-27 21:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add language column to tasks table with unique constraint."""
    # Add language column (nullable for existing tasks and language-agnostic task types)
    op.add_column("tasks", sa.Column("language", sa.String(), nullable=True))

    # Create index on language column for efficient queries
    op.create_index("ix_tasks_language", "tasks", ["language"])

    # Create unique constraint on (video_id, task_type, language)
    # Using COALESCE to handle NULL language values as empty string for uniqueness
    # This ensures only one task per video+type+language combination
    op.execute(
        """
        CREATE UNIQUE INDEX ix_tasks_video_type_language
        ON tasks (video_id, task_type, COALESCE(language, ''))
        """
    )


def downgrade() -> None:
    """Remove language column from tasks table."""
    op.drop_index("ix_tasks_video_type_language", "tasks")
    op.drop_index("ix_tasks_language", "tasks")
    op.drop_column("tasks", "language")
