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

import ibis.expr.types as ibis_types

import bigframes.core.blocks as blocks
import bigframes.operations
import bigframes.series as series


class SeriesMethods:
    def __init__(self, block: blocks.Block):
        assert len(block.value_columns) == 1
        assert len(block.column_labels) == 1
        self._block = block
        self._value_column = self._block.value_columns[0]
        self._name = self._block.column_labels[0]

    @property
    def _value(self) -> ibis_types.Value:
        """Private property to get Ibis expression for the value column."""
        return self._block.expr.get_column(self._value_column)

    def _apply_unary_op(
        self,
        op: bigframes.operations.UnaryOp,
    ) -> series.Series:
        """Applies a unary operator to the series."""
        block, result_id = self._block.apply_unary_op(
            self._value_column, op, result_label=self._name
        )
        return series.Series(block.select_column(result_id))
