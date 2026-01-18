"""Create videos table

Revision ID: 577ef37f8df5
Revises: b3f1edd0a582
Create Date: 2026-01-17 23:40:03.641128

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '577ef37f8df5'
down_revision: Union[str, Sequence[str], None] = 'b3f1edd0a582'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'videos',
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('last_modified', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('video_id'),
        sa.UniqueConstraint('file_path')
    )
    op.create_index('ix_videos_file_path', 'videos', ['file_path'])
    op.create_index('ix_videos_status', 'videos', ['status'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_videos_status', 'videos')
    op.drop_index('ix_videos_file_path', 'videos')
    op.drop_table('videos')
