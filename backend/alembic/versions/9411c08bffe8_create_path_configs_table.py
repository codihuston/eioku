"""Create path_configs table

Revision ID: 9411c08bffe8
Revises: 3b330dae216d
Create Date: 2026-01-18 00:24:01.059327

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9411c08bffe8'
down_revision: Union[str, Sequence[str], None] = '3b330dae216d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'path_configs',
        sa.Column('path_id', sa.String(), nullable=False),
        sa.Column('path', sa.String(), nullable=False),
        sa.Column('recursive', sa.String(), nullable=False),
        sa.Column('added_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('path_id'),
        sa.UniqueConstraint('path')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('path_configs')
