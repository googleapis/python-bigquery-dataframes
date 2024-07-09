# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/expr/operations/arrays.py
from __future__ import annotations

import ibis.expr.datatypes as dt
from ibis.expr.operations.core import Unary


class SafeCastToDatetime(Unary):
    dtype = dt.Timestamp(timezone=None)
