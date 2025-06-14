"""sync timezone columns

Revision ID: 382d285bc5ef
Revises: 66f72f04a1de
Create Date: 2025-06-10 11:18:20.183969

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.engine.reflection import Inspector
from langflow.utils import migration


# revision identifiers, used by Alembic.
revision: str = '382d285bc5ef'
down_revision: Union[str, None] = '66f72f04a1de'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###


def downgrade() -> None:
    conn = op.get_bind()
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###
