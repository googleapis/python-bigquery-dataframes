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

import dataclasses
import functools
import io
import typing

import pandas as pd

import bigframes.core.compile.compiled as compiled
import bigframes.core.compile.concat as concat_impl
import bigframes.core.compile.single_column
import bigframes.core.nodes as nodes

if typing.TYPE_CHECKING:
    import bigframes.core
    import bigframes.session


@dataclasses.dataclass(frozen=True)
class Compiler:
    # Replacements used to implement caching, replacing deep, expensive trees
    # with less expensive trees that leverage previous materializations.
    replacements: tuple[tuple[nodes.BigFrameNode, nodes.BigFrameNode], ...] = ()

    @functools.cached_property
    def _node_lookup(self) -> typing.Mapping[nodes.BigFrameNode, nodes.BigFrameNode]:
        return {a: b for a, b in self.replacements}

    def _maybe_replace_node(self, node: nodes.BigFrameNode) -> nodes.BigFrameNode:
        return self._node_lookup.get(node, node)

    def compile_ordered(self, node: nodes.BigFrameNode) -> compiled.OrderedIR:
        return typing.cast(compiled.OrderedIR, self.compile_node(node, True))

    def compile_unordered(self, node: nodes.BigFrameNode) -> compiled.UnorderedIR:
        return typing.cast(compiled.UnorderedIR, self.compile_node(node, False))

    @functools.cache
    def compile_node(
        self,
        node: nodes.BigFrameNode,
        ordered: bool = True,
    ) -> compiled.UnorderedIR | compiled.OrderedIR:
        """Compile node into CompileArrayValue. Caches result."""
        node = self._maybe_replace_node(node)
        return self._compile_node(node, ordered)

    @functools.singledispatchmethod
    def _compile_node(
        self, node: nodes.BigFrameNode, ordered: bool = True
    ) -> compiled.UnorderedIR:
        """Defines transformation but isn't cached, always use compile_node instead"""
        raise ValueError(f"Can't compile unrecognized node: {node}")

    @_compile_node.register
    def compile_join(self, node: nodes.JoinNode, ordered: bool = True):
        if ordered:
            left_ordered = self.compile_ordered(node.left_child)
            right_ordered = self.compile_ordered(node.right_child)
            return bigframes.core.compile.single_column.join_by_column_ordered(
                left_ordered,
                node.left_column_ids,
                right_ordered,
                node.right_column_ids,
                how=node.how,
                allow_row_identity_join=node.allow_row_identity_join,
            )
        else:
            left_unordered = self.compile_unordered(node.left_child)
            right_unordered = self.compile_unordered(node.right_child)
            return bigframes.core.compile.single_column.join_by_column_unordered(
                left_unordered,
                node.left_column_ids,
                right_unordered,
                node.right_column_ids,
                how=node.how,
                allow_row_identity_join=node.allow_row_identity_join,
            )

    @_compile_node.register
    def compile_select(self, node: nodes.SelectNode, ordered: bool = True):
        return self.compile_node(node.child, ordered).select_columns(node.column_ids)

    @_compile_node.register
    def compile_drop(self, node: nodes.DropColumnsNode, ordered: bool = True):
        return self.compile_node(node.child, ordered).drop_columns(node.columns)

    @_compile_node.register
    def compile_readlocal(self, node: nodes.ReadLocalNode, ordered: bool = True):
        array_as_pd = pd.read_feather(io.BytesIO(node.feather_bytes))
        ordered_ir = compiled.OrderedIR.from_pandas(array_as_pd)
        if ordered:
            return ordered_ir
        else:
            return ordered_ir.to_unordered()

    @_compile_node.register
    def compile_readgbq(self, node: nodes.ReadGbqNode, ordered: bool = True):
        if ordered:
            return compiled.OrderedIR(
                node.table,
                node.columns,
                node.hidden_ordering_columns,
                node.ordering,
            )
        else:
            return compiled.UnorderedIR(
                node.table,
                node.columns,
            )

    @_compile_node.register
    def compile_promote_offsets(
        self, node: nodes.PromoteOffsetsNode, ordered: bool = True
    ):
        result = self.compile_ordered(node.child).promote_offsets(node.col_id)
        return result if ordered else result.to_unordered()

    @_compile_node.register
    def compile_filter(self, node: nodes.FilterNode, ordered: bool = True):
        return self.compile_node(node.child, ordered).filter(
            node.predicate_id, node.keep_null
        )

    @_compile_node.register
    def compile_orderby(self, node: nodes.OrderByNode, ordered: bool = True):
        if ordered:
            return self.compile_ordered(node.child).order_by(node.by)
        else:
            return self.compile_unordered(node.child)

    @_compile_node.register
    def compile_reversed(self, node: nodes.ReversedNode, ordered: bool = True):
        if ordered:
            return self.compile_ordered(node.child).reversed()
        else:
            return self.compile_unordered(node.child)

    @_compile_node.register
    def compile_project_unary(
        self, node: nodes.ProjectUnaryOpNode, ordered: bool = True
    ):
        return self.compile_node(node.child, ordered).project_unary_op(
            node.input_id, node.op, node.output_id
        )

    @_compile_node.register
    def compile_project_binary(
        self, node: nodes.ProjectBinaryOpNode, ordered: bool = True
    ):
        return self.compile_node(node.child, ordered).project_binary_op(
            node.left_input_id, node.right_input_id, node.op, node.output_id
        )

    @_compile_node.register
    def compile_project_ternary(
        self, node: nodes.ProjectTernaryOpNode, ordered: bool = True
    ):
        return self.compile_node(node.child, ordered).project_ternary_op(
            node.input_id1, node.input_id2, node.input_id3, node.op, node.output_id
        )

    @_compile_node.register
    def compile_concat(self, node: nodes.ConcatNode, ordered: bool = True):
        if ordered:
            compiled_ordered = [self.compile_ordered(node) for node in node.children]
            return concat_impl.concat_ordered(compiled_ordered)
        else:
            compiled_unordered = [
                self.compile_unordered(node) for node in node.children
            ]
            return concat_impl.concat_unordered(compiled_unordered)

    @_compile_node.register
    def compile_aggregate(self, node: nodes.AggregateNode, ordered: bool = True):
        result = self.compile_unordered(node.child).aggregate(
            node.aggregations, node.by_column_ids, node.dropna
        )
        return result if ordered else result.to_unordered()

    @_compile_node.register
    def compile_corr(self, node: nodes.CorrNode, ordered: bool = True):
        result = self.compile_unordered(node.child).corr_aggregate(
            node.corr_aggregations
        )
        return result if ordered else result.to_unordered()

    @_compile_node.register
    def compile_window(self, node: nodes.WindowOpNode, ordered: bool = True):
        result = self.compile_ordered(node.child).project_window_op(
            node.column_name,
            node.op,
            node.window_spec,
            node.output_name,
            never_skip_nulls=node.never_skip_nulls,
            skip_reproject_unsafe=node.skip_reproject_unsafe,
        )
        return result if ordered else result.to_unordered()

    @_compile_node.register
    def compile_reproject(self, node: nodes.ReprojectOpNode, ordered: bool = True):
        return self.compile_node(node.child, ordered)._reproject_to_table()

    @_compile_node.register
    def compile_unpivot(self, node: nodes.UnpivotNode, ordered: bool = True):
        return self.compile_node(node.child, ordered).unpivot(
            node.row_labels,
            node.unpivot_columns,
            passthrough_columns=node.passthrough_columns,
            index_col_ids=node.index_col_ids,
            dtype=node.dtype,
            how=node.how,
        )

    @_compile_node.register
    def compile_assign(self, node: nodes.AssignNode, ordered: bool = True):
        return self.compile_node(node.child, ordered).assign(
            node.source_id, node.destination_id
        )

    @_compile_node.register
    def compile_assign_constant(
        self, node: nodes.AssignConstantNode, ordered: bool = True
    ):
        return self.compile_node(node.child, ordered).assign_constant(
            node.destination_id, node.value, node.dtype
        )

    @_compile_node.register
    def compiler_random_sample(
        self, node: nodes.RandomSampleNode, ordered: bool = True
    ):
        return self.compile_node(node.child, ordered)._uniform_sampling(node.fraction)
