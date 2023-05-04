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

import bigframes.operations as ops
import bigframes.operations.base
import bigframes.series as series


class StringMethods(bigframes.operations.base.SeriesMethods):
    """Methods that act on a string Series."""

    def find(self, sub, start=None, end=None) -> series.Series:
        """Return the position of the first occurence of substring."""
        return self._apply_unary_op(ops.FindOp(sub, start, end))

    def len(self) -> series.Series:
        """Compute the length of each string."""
        return self._apply_unary_op(ops.len_op)

    def lower(self) -> series.Series:
        """Convert strings in the Series to lowercase."""
        return self._apply_unary_op(ops.lower_op)

    def reverse(self) -> series.Series:
        """Reverse strings in the Series."""
        return self._apply_unary_op(ops.reverse_op)

    def slice(self, start=None, stop=None) -> series.Series:
        """Slice substrings from each element in the Series."""
        return self._apply_unary_op(ops.SliceOp(start, stop))

    def strip(self) -> series.Series:
        """Removes whitespace characters from the beginning and end of each string in the Series."""
        return self._apply_unary_op(ops.strip_op)

    def upper(self) -> series.Series:
        """Convert strings in the Series to uppercase."""
        return self._apply_unary_op(ops.upper_op)

    def isnumeric(self) -> series.Series:
        """Check whether all characters in each string are numeric."""
        return self._apply_unary_op(ops.isnumeric_op)
