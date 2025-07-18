from __future__ import annotations

from bigframes_vendored.ibis.common.typing import VarTuple  # noqa: TCH001
import bigframes_vendored.ibis.expr.datashape as ds
import bigframes_vendored.ibis.expr.datatypes as dt
from bigframes_vendored.ibis.expr.operations.core import Value
from public import public


@public
class RawSql(Value):
    sql: str
    dtype: dt.DataType
    inputs: VarTuple[Value] = ()

    shape = ds.columnar
