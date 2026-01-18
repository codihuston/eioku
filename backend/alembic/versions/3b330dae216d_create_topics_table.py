"""Create topics table

Revision ID: 3b330dae216d
Revises: 2fb94f359c6b
Create Date: 2026-01-18 00:23:06.920939

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3b330dae216d'
down_revision: Union[str, Sequence[str], None] = '2fb94f359c6b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'topics',
        sa.Column('topic_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        sa.Column('keywords', sa.JSON(), nullable=False),
        sa.Column('relevance_score', sa.Float(), nullable=False),
        sa.Column('timestamps', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id']),
        sa.PrimaryKeyConstraint('topic_id')
    )
    op.create_index('ix_topics_video_id', 'topics', ['video_id'])
    op.create_index('ix_topics_label', 'topics', ['label'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_topics_label', 'topics')
    op.drop_index('ix_topics_video_id', 'topics')
    op.drop_table('topics')
