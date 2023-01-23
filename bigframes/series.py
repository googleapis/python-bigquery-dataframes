"""Series is a 1 dimensional data structure."""

from __future__ import annotations
import pandas

try:
    from ibis.expr.types import Table, Value
except ImportError:
    from ibis.expr.types import TableExpr as Table
    from ibis.expr.types import ValueExpr as Value


class Series:
    """A 1D data structure, representing data and deferred computation.

    .. warning::
        This constructor is **private**. Use a public method such as
        ``DataFrame[column_name]`` to construct a Series.
    """

    def __init__(self, table: Table, value: Value):
        super().__init__()
        self._table = table
        self._value = value

    def _to_ibis_expr(self):
        return self._table.select(self._value).get_column(self._value.get_name())

    def compute(self) -> pandas.Series:
        """Executes deferred operations and downloads the results."""
        value = self._to_ibis_expr()
        return value.execute()

    def lower(self) -> "Series":
        """Convert strings in the Series to lowercase."""
        return Series(self._table, self._value.lower().name(self._value.get_name()))

    def upper(self) -> "Series":
        """Convert strings in the Series to uppercase."""
        return Series(self._table, self._value.upper().name(self._value.get_name()))

    def __add__(self, other: float | int | Series | pandas.Series) -> Series:
        if isinstance(other, Series):
            return Series(
                self._table,
                self._value.__add__(other._value).name(self._value.get_name()),
            )
        else:
            return Series(
                self._table, self._value.__add__(other).name(self._value.get_name())
            )

    def reverse(self) -> "Series":
        """Reverse strings in the Series."""
        return Series(self._table, self._value.reverse().name(self._value.get_name()))
