"""Create transcriptions table

Revision ID: 40b2ad679602
Revises: 577ef37f8df5
Create Date: 2026-01-18 00:15:22.179427

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '40b2ad679602'
down_revision: Union[str, Sequence[str], None] = '577ef37f8df5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'transcriptions',
        sa.Column('segment_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('start', sa.Float(), nullable=False),
        sa.Column('end', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('speaker', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id']),
        sa.PrimaryKeyConstraint('segment_id')
    )
    op.create_index('ix_transcriptions_video_id', 'transcriptions', ['video_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_transcriptions_video_id', 'transcriptions')
    op.drop_table('transcriptions')
