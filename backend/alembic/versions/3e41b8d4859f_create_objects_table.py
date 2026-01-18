"""Create objects table

Revision ID: 3e41b8d4859f
Revises: afc3fae0d4ef
Create Date: 2026-01-18 00:18:04.213897

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3e41b8d4859f'
down_revision: Union[str, Sequence[str], None] = 'afc3fae0d4ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'objects',
        sa.Column('object_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        sa.Column('timestamps', sa.JSON(), nullable=False),
        sa.Column('bounding_boxes', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id']),
        sa.PrimaryKeyConstraint('object_id')
    )
    op.create_index('ix_objects_video_id', 'objects', ['video_id'])
    op.create_index('ix_objects_label', 'objects', ['label'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_objects_label', 'objects')
    op.drop_index('ix_objects_video_id', 'objects')
    op.drop_table('objects')
