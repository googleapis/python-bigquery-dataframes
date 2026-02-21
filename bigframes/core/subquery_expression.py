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

import abc
import dataclasses
import functools
import itertools
import typing
from typing import Callable, Mapping, Tuple

from bigframes import dtypes
from bigframes.core import bigframe_node, expression
import bigframes.core.identifiers as ids


@dataclasses.dataclass(frozen=True)
class SubqueryExpression(expression.Expression):
    """Represents windowing or aggregation over a column."""

    subquery: bigframe_node.BigFrameNode

    @property
    def column_references(self) -> typing.Tuple[ids.ColumnId, ...]:
        return tuple(
            itertools.chain.from_iterable(
                map(lambda x: x.column_references, self.inputs)
            )
        )

    @functools.cached_property
    def is_resolved(self) -> bool:
        return False

    @functools.cached_property
    def output_type(self) -> dtypes.ExpressionType:
        raise ValueError("Subquery has no output type.")

    @property
    @abc.abstractmethod
    def inputs(
        self,
    ) -> typing.Tuple[expression.Expression, ...]:
        ...

    @property
    def children(self) -> Tuple[expression.Expression, ...]:
        return self.inputs

    @property
    def free_variables(self) -> typing.Tuple[str, ...]:
        return tuple(
            itertools.chain.from_iterable(map(lambda x: x.free_variables, self.inputs))
        )

    @property
    def is_const(self) -> bool:
        return all(child.is_const for child in self.inputs)

    @functools.cached_property
    def is_scalar_expr(self) -> bool:
        return False

    @abc.abstractmethod
    def replace_args(self, *arg) -> SubqueryExpression:
        ...

    def transform_children(
        self, t: Callable[[expression.Expression], expression.Expression]
    ) -> SubqueryExpression:
        return self.replace_args(*(t(arg) for arg in self.inputs))

    def bind_variables(
        self,
        bindings: Mapping[str, expression.Expression],
        allow_partial_bindings: bool = False,
    ) -> SubqueryExpression:
        return self.transform_children(
            lambda x: x.bind_variables(bindings, allow_partial_bindings)
        )

    def bind_refs(
        self,
        bindings: Mapping[ids.ColumnId, expression.Expression],
        allow_partial_bindings: bool = False,
    ) -> SubqueryExpression:
        return self.transform_children(
            lambda x: x.bind_refs(bindings, allow_partial_bindings)
        )
