# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/expr/operations/json.py
from __future__ import annotations

import ibis.common.typing as ibis_typing
import ibis.expr.datatypes as dt
import ibis.expr.operations.core as ibis_ops_core
import ibis.expr.rules as rlz


class ToJsonString(ibis_ops_core.Unary):
    dtype = dt.string


class JSONSet(ibis_ops_core.Unary):
    json_path_value_pairs: ibis_typing.VarTuple[
        ibis_typing.VarTuple[ibis_ops_core.Value[dt.Any]]
    ]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")
