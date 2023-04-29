from __future__ import annotations

from enum import Enum
import math
import typing
from typing import Optional, Sequence

import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types

# TODO(tbergeron): Encode more efficiently
ORDERING_ID_STRING_BASE: int = 10
# Sufficient to store any value up to 2^63
DEFAULT_ORDERING_ID_LENGTH: int = math.ceil(63 * math.log(2, ORDERING_ID_STRING_BASE))


class OrderingDirection(Enum):
    ASC = 1
    DESC = 2


class ExpressionOrdering:
    """Immutable object that holds information about the ordering of rows in a BigFrames expression."""

    def __init__(
        self,
        ordering_value_columns: Optional[Sequence[str]] = None,
        ordering_id_column: Optional[str] = None,
        is_sequential: bool = False,
        ascending: bool = True,
        ordering_encoding_size=DEFAULT_ORDERING_ID_LENGTH,
    ):
        # TODO(tbergeron): Allow flag to reverse ordering
        self._ordering_value_columns = (
            tuple(ordering_value_columns) if ordering_value_columns else ()
        )
        self._ordering_id_column = ordering_id_column
        self._is_sequential = is_sequential
        self._ascending = ascending
        # Encoding size must be tracked in order to know what how to combine ordering ids across tables (eg how much to pad when combining different length).
        # Also will be needed to determine when length is too large and need to compact ordering id with a ROW_NUMBER operation.
        self._ordering_encoding_size = ordering_encoding_size

    def with_is_sequential(self, is_sequential: bool):
        """Create a copy that is marked as non-sequential, this is useful when filtering, but not sorting, an expression."""
        return ExpressionOrdering(
            self._ordering_value_columns,
            self._ordering_id_column,
            is_sequential,
            ordering_encoding_size=self._ordering_encoding_size,
        )

    def with_ordering_columns(
        self, ordering_value_columns: Optional[Sequence[str]] = None, ascending=True
    ):
        """Creates a new ordering that preserves ordering id, but replaces ordering value column list."""
        return ExpressionOrdering(
            ordering_value_columns,
            self._ordering_id_column,
            is_sequential=False,
            ascending=ascending,
            ordering_encoding_size=self._ordering_encoding_size,
        )

    def with_ordering_id(self, ordering_id: str):
        """Creates a new ordering that preserves other properties, but with a different ordering id. Useful when reprojecting ordering for implicit joins."""
        return ExpressionOrdering(
            self._ordering_value_columns,
            ordering_id,
            is_sequential=self.is_sequential,
            ascending=self.is_ascending,
            ordering_encoding_size=self._ordering_encoding_size,
        )

    def with_reverse(self):
        """Reverses the ordering."""
        return ExpressionOrdering(
            self._ordering_value_columns,
            self._ordering_id_column,
            is_sequential=False,
            ascending=(not self._ascending),
            ordering_encoding_size=self._ordering_encoding_size,
        )

    @property
    def is_sequential(self) -> bool:
        return self._is_sequential

    @property
    def is_ascending(self) -> bool:
        return self._ascending

    @property
    def ordering_value_columns(self) -> Sequence[str]:
        return self._ordering_value_columns

    @property
    def ordering_id(self) -> Optional[str]:
        return self._ordering_id_column

    @property
    def ordering_id_encoding_size(self) -> Optional[str]:
        """Length of the ordering id when encoded in string form."""
        return self._ordering_encoding_size

    @property
    def order_id_defined(self) -> bool:
        """True if ordering is fully defined in ascending order by its ordering id."""
        return bool(
            self._ordering_id_column
            and (not self._ordering_value_columns)
            and self.is_ascending
        )

    @property
    def all_ordering_columns(self) -> Sequence[str]:
        return (
            list(self._ordering_value_columns)
            if self._ordering_id_column is None
            else [*self._ordering_value_columns, self._ordering_id_column]
        )


def stringify_order_id(
    order_id: ibis_types.Value, length: int = DEFAULT_ORDERING_ID_LENGTH
) -> ibis_types.StringValue:
    """Converts an order id value to string if it is not already a string. MUST produced fixed-length strings."""
    if order_id.type().is_int64():
        # This is very inefficient encoding base-10 string uses only 10 characters per byte(out of 256 bit combinations)
        # Furthermore, if know tighter bounds on order id are known, can produce smaller strings.
        # 19 characters chosen as it can represent any positive Int64 in base-10
        # For missing values, ":" * 19 is used as it is larger than any other value this function produces, so null values will be last.
        string_order_id = (
            typing.cast(
                ibis_types.StringValue,
                typing.cast(ibis_types.IntegerValue, order_id).cast(ibis_dtypes.string),
            )
            .lpad(length, "0")
            .fillna(ibis_types.literal(":" * length))
        )
    else:
        string_order_id = (
            typing.cast(ibis_types.StringValue, order_id)
            .lpad(length, "0")
            .fillna(ibis_types.literal(":" * length))
        )
    return typing.cast(ibis_types.StringValue, string_order_id)
