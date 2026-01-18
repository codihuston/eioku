"""Create faces table

Revision ID: 2fb94f359c6b
Revises: 3e41b8d4859f
Create Date: 2026-01-18 00:21:00.251989

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2fb94f359c6b'
down_revision: Union[str, Sequence[str], None] = '3e41b8d4859f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'faces',
        sa.Column('face_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('person_id', sa.String(), nullable=True),
        sa.Column('timestamps', sa.JSON(), nullable=False),
        sa.Column('bounding_boxes', sa.JSON(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id']),
        sa.PrimaryKeyConstraint('face_id')
    )
    op.create_index('ix_faces_video_id', 'faces', ['video_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_faces_video_id', 'faces')
    op.drop_table('faces')
