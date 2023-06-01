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

"""Classes that handle row labels, ordering, and implicit joins."""

from __future__ import annotations

import typing
from typing import Callable, Tuple

import bigframes.core as core
import bigframes.core.blocks as blocks
import bigframes.core.joins as joins


class ImplicitJoiner:
    """Allow implicit joins without row labels on related table expressions."""

    def __init__(self, block: blocks.Block, name: typing.Optional[str] = None):
        self._block = block
        self._name = name

    def copy(self) -> ImplicitJoiner:
        """Make a copy of this object."""
        # TODO(swast): Should this make a copy of block?
        return ImplicitJoiner(self._block, self._name)

    @property
    def _expr(self) -> core.BigFramesExpr:
        return self._block.expr

    @property
    def name(self) -> typing.Optional[str]:
        """Name of the Index."""
        # This introduces a level of indirection over Ibis to allow for more
        # accurate pandas behavior (such as allowing for unnamed or
        # non-uniquely named objects) without breaking SQL generation.
        return self._name

    @name.setter
    def name(self, value: typing.Optional[str]):
        self._name = value

    # TODO(swast): In pandas, "left_indexer" and "right_indexer" are numpy
    # arrays that indicate where the rows line up. Do we want to wrap ibis to
    # emulate arrays? How might this change if we're doing a real join on the
    # respective table expressions? See:
    # https://pandas.pydata.org/docs/reference/api/pandas.Index.get_indexer.html
    def join(
        self,
        other: ImplicitJoiner,
        *,
        how="left",
    ) -> Tuple[ImplicitJoiner, Tuple[Callable[[str], str], Callable[[str], str]],]:
        """Compute join_index and indexers to conform data structures to the new index."""
        joined_expr, (get_column_left, get_column_right) = joins.join_by_row_identity(
            self._expr, other._expr, how=how
        )
        block = blocks.Block(
            joined_expr,
            column_labels=[*self._block.column_labels, *other._block.column_labels],
        )
        return ImplicitJoiner(block, name=self.name), (
            get_column_left,
            get_column_right,
        )
