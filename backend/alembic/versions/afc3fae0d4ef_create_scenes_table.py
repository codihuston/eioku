"""Create scenes table

Revision ID: afc3fae0d4ef
Revises: 40b2ad679602
Create Date: 2026-01-18 00:16:11.236766

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'afc3fae0d4ef'
down_revision: Union[str, Sequence[str], None] = '40b2ad679602'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'scenes',
        sa.Column('scene_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('scene', sa.Integer(), nullable=False),
        sa.Column('start', sa.Float(), nullable=False),
        sa.Column('end', sa.Float(), nullable=False),
        sa.Column('thumbnail_path', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id']),
        sa.PrimaryKeyConstraint('scene_id')
    )
    op.create_index('ix_scenes_video_id', 'scenes', ['video_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_scenes_video_id', 'scenes')
    op.drop_table('scenes')
