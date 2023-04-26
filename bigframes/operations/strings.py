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

from __future__ import annotations

import typing

import ibis.expr.types as ibis_types

import bigframes.operations as ops
import bigframes.operations.base
import bigframes.series as series


class StringMethods(bigframes.operations.base.SeriesMethods):
    """Methods that act on a string Series."""

    def find(self, sub, start=None, end=None) -> series.Series:
        """Return the position of the first occurence of substring."""

        def find_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).find(sub, start, end)

        return self._apply_unary_op(find_op)

    def len(self) -> series.Series:
        """Compute the length of each string."""

        return self._apply_unary_op(ops.len_op)

    def lower(self) -> series.Series:
        """Convert strings in the Series to lowercase."""

        def lower_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).lower()

        return self._apply_unary_op(lower_op)

    def reverse(self) -> series.Series:
        """Reverse strings in the Series."""
        return self._apply_unary_op(ops.reverse_op)

    def slice(self, start=None, stop=None) -> series.Series:
        """Slice substrings from each element in the Series."""

        def slice_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x)[start:stop]

        return self._apply_unary_op(slice_op)

    def strip(self) -> series.Series:
        """Removes whitespace characters from the beginning and end of each string in the Series."""

        def strip_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).strip()

        return self._apply_unary_op(strip_op)

    def upper(self) -> series.Series:
        """Convert strings in the Series to uppercase."""

        def upper_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).upper()

        return self._apply_unary_op(upper_op)
