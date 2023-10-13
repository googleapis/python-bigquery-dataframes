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

from dataclasses import dataclass, field
import typing
from typing import Optional, Tuple

import pandas

import bigframes.core.guid
from bigframes.core.ordering import OrderingColumnReference
import bigframes.core.window_spec as window
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops

if typing.TYPE_CHECKING:
    import ibis.expr.types as ibis_types

    import bigframes.core.ordering as orderings
    import bigframes.session


@dataclass(frozen=True)
class BigFrameNode:
    pass

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return tuple([])

    @property
    def sessions(self):
        sessions = []
        for child in self.child_nodes:
            sessions.extend(child.sessions)
        return sessions


@dataclass(frozen=True)
class UnaryNode(BigFrameNode):
    child: BigFrameNode

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return (self.child,)


@dataclass(frozen=True)
class JoinNode(BigFrameNode):
    left_child: BigFrameNode
    right_child: BigFrameNode
    left_column_ids: typing.Tuple[str, ...]
    right_column_ids: typing.Tuple[str, ...]
    how: typing.Literal[
        "inner",
        "left",
        "outer",
        "right",
    ]
    allow_row_identity_join: bool = True

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return (self.left_child, self.right_child)


@dataclass(frozen=True)
class ConcatNode(BigFrameNode):
    children: Tuple[BigFrameNode, ...]

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return self.children


# Input Nodex
@dataclass(frozen=True)
class ReadLocalNode(BigFrameNode):
    # column major
    local_array: typing.Tuple[typing.Tuple[typing.Hashable, ...], ...]
    column_ids: typing.Tuple[str, ...]


# TODO: Refactor to take raw gbq object reference
@dataclass(frozen=True)
class ReadGbqNode(BigFrameNode):
    table: ibis_types.Table = field()
    session: bigframes.session.Session = field()
    columns: Tuple[ibis_types.Value, ...] = field()
    hidden_ordering_columns: Tuple[ibis_types.Value, ...] = field()
    ordering: orderings.ExpressionOrdering = field()

    @property
    def sessions(self):
        return (self.session,)


# Unary nodes
@dataclass(frozen=True)
class DropColumnsNode(UnaryNode):
    columns: Tuple[str, ...]


@dataclass(frozen=True)
class PromoteOffsetsNode(UnaryNode):
    col_id: str


@dataclass(frozen=True)
class FilterNode(UnaryNode):
    predicate_id: str
    keep_null: bool = False


@dataclass(frozen=True)
class OrderByNode(UnaryNode):
    by: Tuple[OrderingColumnReference, ...]
    stable: bool = False


@dataclass(frozen=True)
class ReversedNode(UnaryNode):
    pass


@dataclass(frozen=True)
class SelectNode(UnaryNode):
    column_ids: typing.Tuple[str, ...]


@dataclass(frozen=True)
class ProjectUnaryOpNode(UnaryNode):
    input_id: str
    op: ops.UnaryOp
    output_id: Optional[str] = None


@dataclass(frozen=True)
class ProjectBinaryOpNode(UnaryNode):
    left_input_id: str
    right_input_id: str
    op: ops.BinaryOp
    output_id: str


@dataclass(frozen=True)
class ProjectTernaryOpNode(UnaryNode):
    input_id1: str
    input_id2: str
    input_id3: str
    op: ops.TernaryOp
    output_id: str


@dataclass(frozen=True)
class AggregateNode(UnaryNode):
    aggregations: typing.Tuple[typing.Tuple[str, agg_ops.AggregateOp, str], ...]
    by_column_ids: typing.Tuple[str, ...] = tuple([])
    dropna: bool = True


# TODO: Unify into aggregate
@dataclass(frozen=True)
class CorrNode(UnaryNode):
    corr_aggregations: typing.Tuple[typing.Tuple[str, str, str], ...]


@dataclass(frozen=True)
class WindowOpNode(UnaryNode):
    column_name: str
    op: agg_ops.WindowOp
    window_spec: window.WindowSpec
    output_name: typing.Optional[str] = None
    never_skip_nulls: bool = False
    skip_reproject_unsafe: bool = False


@dataclass(frozen=True)
class ReprojectOpNode(UnaryNode):
    pass


@dataclass(frozen=True)
class UnpivotNode(UnaryNode):
    row_labels: typing.Tuple[typing.Hashable, ...]
    unpivot_columns: typing.Tuple[
        typing.Tuple[str, typing.Tuple[typing.Optional[str], ...]], ...
    ]
    passthrough_columns: typing.Tuple[str, ...] = ()
    index_col_ids: typing.Tuple[str, ...] = ("index",)
    dtype: typing.Union[
        bigframes.dtypes.Dtype, typing.Sequence[bigframes.dtypes.Dtype]
    ] = (pandas.Float64Dtype(),)
    how: typing.Literal["left", "right"] = "left"


@dataclass(frozen=True)
class AssignNode(UnaryNode):
    source_id: str
    destination_id: str


@dataclass(frozen=True)
class AssignConstantNode(UnaryNode):
    destination_id: str
    value: typing.Hashable
    dtype: typing.Optional[bigframes.dtypes.Dtype]


@dataclass(frozen=True)
class RandomSampleNode(UnaryNode):
    fraction: float

    @property
    def deterministic(self) -> bool:
        return False
