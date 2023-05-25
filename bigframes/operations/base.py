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

from typing import Optional

import ibis.expr.types as ibis_types

import bigframes.core.blocks as blocks
import bigframes.operations
import bigframes.series as series


class SeriesMethods:
    def __init__(
        self,
        block: blocks.Block,
        value_column: str,
        *,
        name: Optional[str] = None,
    ):
        self._block = block
        self._value_column = value_column
        self._name = name

    @property
    def _viewed_block(self) -> blocks.Block:
        """Gets a copy of block after any views have been applied. Mutations to this
        copy do not affect any existing series/dataframes."""
        return self._block.copy()

    @property
    def _value(self) -> ibis_types.Value:
        """Private property to get Ibis expression for the value column."""
        return self._block.expr.get_column(self._value_column)

    def _apply_unary_op(
        self,
        op: bigframes.operations.UnaryOp,
    ) -> series.Series:
        """Applies a unary operator to the series."""
        block = self._viewed_block
        block.apply_unary_op(self._value_column, op)
        return series.Series(
            block,
            self._value_column,
            name=self._name,
        )
