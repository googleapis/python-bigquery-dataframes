# Contains code from https://github.com/ibis-project/ibis/blob/9.2.0/ibis/expr/operations/reductions.py

"""Reduction operations."""

from __future__ import annotations

from typing import Optional

import bigframes_vendored.ibis.common.annotations as ibis_annotations
from bigframes_vendored.ibis.common.typing import VarTuple
from bigframes_vendored.ibis.expr.operations.core import Column, Value
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from public import public


@public
class Reduction(Value):
    """Base class for reduction operations."""

    shape = ds.scalar


# TODO(kszucs): all reductions all filterable so we could remove Filterable
class Filterable(Value):
    where: Optional[Value[dt.Boolean]] = None


class ArrayAggregate(Filterable, Reduction):
    """
    Collects the elements of this expression into an ordered array. Similar to
    the ibis `ArrayCollect`, but adds `order_by_*` and `distinct_only` parameters.
    """

    arg: Column
    order_by: VarTuple[Value] = ()

    @ibis_annotations.attribute
    def dtype(self):
        return dt.Array(self.arg.dtype)


__all__ = ["ArrayAggregate"]
