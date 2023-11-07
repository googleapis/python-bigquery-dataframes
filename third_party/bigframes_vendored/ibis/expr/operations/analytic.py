# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/expr/operations/analytic.py

from __future__ import annotations

import ibis.expr.datatypes as dt
from ibis.expr.operations.analytic import Analytic
from ibis.expr.operations.core import Column
import ibis.expr.rules as rlz


class FirstNonNullValue(Analytic):
    """Retrieve the first element."""

    arg: Column[dt.Any]
    output_dtype = rlz.dtype_like("arg")


class LastNonNullValue(Analytic):
    """Retrieve the last element."""

    arg: Column[dt.Any]
    output_dtype = rlz.dtype_like("arg")


__all__ = [
    "FirstNonNullValue",
    "LastNonNullValue",
]
