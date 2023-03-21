"""BigFrames top level APIs."""
from typing import Iterable

import ibis

import bigframes.core
from bigframes.core import blocks
from bigframes.dataframe import DataFrame


def concat(objs: Iterable[DataFrame]) -> DataFrame:
    """Concatenate BigFrames objects along rows.

    Note: currently only supports DataFrames with identical schemas (including index columns).
    """
    # TODO(garrettwu): Figure out how to support DataFrames with different schema, or emit appropriate error message.
    objs = list(objs)
    block_0 = objs[0]._block
    tables = [obj._block.expr.to_ibis_expr() for obj in objs]
    expr = bigframes.core.BigFramesExpr(block_0.expr._session, ibis.union(*tables))

    block = blocks.Block(expr, block_0.index_columns)
    return DataFrame(block)
