"""Create tasks table

Revision ID: 12af1b7fb041
Revises: 9411c08bffe8
Create Date: 2026-01-18 00:24:53.902508

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '12af1b7fb041'
down_revision: Union[str, Sequence[str], None] = '9411c08bffe8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'tasks',
        sa.Column('task_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('task_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('dependencies', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id']),
        sa.PrimaryKeyConstraint('task_id')
    )
    op.create_index('ix_tasks_video_id', 'tasks', ['video_id'])
    op.create_index('ix_tasks_task_type', 'tasks', ['task_type'])
    op.create_index('ix_tasks_status', 'tasks', ['status'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_tasks_status', 'tasks')
    op.drop_index('ix_tasks_task_type', 'tasks')
    op.drop_index('ix_tasks_video_id', 'tasks')
    op.drop_table('tasks')
