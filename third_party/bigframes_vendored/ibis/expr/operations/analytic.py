# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/expr/operations/analytic.py

from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.operations.analytic as ibis_ops_analytic
import ibis.expr.operations.core as ibis_ops_core
import ibis.expr.rules as rlz


class FirstNonNullValue(ibis_ops_analytic.Analytic):
    """Retrieve the first element."""

    arg: ibis_ops_core.Column[dt.Any]
    dtype = rlz.dtype_like("arg")


class LastNonNullValue(ibis_ops_analytic.Analytic):
    """Retrieve the last element."""

    arg: ibis_ops_core.Column[dt.Any]
    dtype = rlz.dtype_like("arg")


__all__ = [
    "FirstNonNullValue",
    "LastNonNullValue",
]
