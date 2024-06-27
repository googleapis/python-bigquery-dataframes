# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/expr/operations/json.py
from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.operations.core as ibis_ops_core
import ibis.expr.rules as rlz


class ToJsonString(ibis_ops_core.Unary):
    dtype = dt.string


class JSONSet(ibis_ops_core.Unary):
    json_value: ibis_ops_core.Value[dt.Any]
    json_path: ibis_ops_core.Value[dt.String]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")
