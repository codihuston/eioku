"""Add file_hash to videos table

Revision ID: 4022c4cef20d
Revises: 12af1b7fb041
Create Date: 2026-01-18 01:17:28.916273

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4022c4cef20d'
down_revision: Union[str, Sequence[str], None] = '12af1b7fb041'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('videos', sa.Column('file_hash', sa.String(), nullable=True))
    op.create_index('ix_videos_file_hash', 'videos', ['file_hash'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_videos_file_hash', 'videos')
    op.drop_column('videos', 'file_hash')
