# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BigFrames top level APIs."""
from typing import Iterable

import ibis

import bigframes.core
from bigframes.core import blocks
from bigframes.dataframe import DataFrame


def concat(objs: Iterable[DataFrame]) -> DataFrame:
    """Concatenate BigFrames objects along rows.

    Note: currently only supports DataFrames with identical schemas (including index columns) and column names.
    """
    # TODO(garrettwu): Figure out how to support DataFrames with different schema, or emit appropriate error message.
    objs = list(objs)
    block_0 = objs[0]._block
    tables = [obj._block.expr.to_ibis_expr() for obj in objs]
    expr = bigframes.core.BigFramesExpr(block_0.expr._session, ibis.union(*tables))

    block = blocks.Block(expr, block_0.index_columns)
    return DataFrame(block.index, objs[0]._col_names)
