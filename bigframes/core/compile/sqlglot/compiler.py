# Copyright 2025 Google LLC
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
import itertools
import typing

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import sqlglot as sg
import sqlglot.expressions as sge

import bigframes.core
from bigframes.core import utils
import bigframes.core.compile.sqlglot.scalar_op_compiler as scalar_op_compiler
import bigframes.core.compile.sqlglot.sqlglot_types as sgt
import bigframes.core.expression as ex
import bigframes.core.guid as guid
import bigframes.core.identifiers as ids
import bigframes.core.nodes as nodes
import bigframes.core.ordering
import bigframes.core.rewrite
import bigframes.core.rewrite as rewrites
import bigframes.dtypes as dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops


@dataclasses.dataclass(frozen=True)
class SQLGlotCompiler:
    """
    Compiles BigFrameNode to SQLGlot expression tree recursively.
    """

    # In strict mode, ordering will always be deterministic
    # In unstrict mode, ordering from ReadTable or after joins may be ambiguous to improve query performance.
    strict: bool = True
    # Whether to always quote identifiers.
    quoted: bool = True
    # TODO: the way how the scalar operation compiles stop the non-recursive compiler.
    # Define scalar compiler for converting bigframes expressions to sqlglot expressions.
    scalar_op_compiler = scalar_op_compiler.SQLGlotScalarOpCompiler()

    # Creates sequential IDs with separate counters for each prefix (e.g., "t", "c").
    # ID sequences are unique per instance of this class.
    uid_generator = guid.SequentialUIDGenerator()

    # TODO: add BigQuery Dialect
    def compile_sql(
        self,
        node: nodes.BigFrameNode,
        ordered: bool,
        limit: typing.Optional[int] = None,
    ) -> sge.Select:
        # later steps might add ids, so snapshot before those steps.
        output_ids = node.schema.names
        if ordered:
            # Need to do this before replacing unsupported ops, as that will rewrite slice ops
            node, pulled_up_limit = rewrites.pullup_limit_from_slice(node)
            if (pulled_up_limit is not None) and (
                (limit is None) or limit > pulled_up_limit
            ):
                limit = pulled_up_limit

        node = self._replace_unsupported_ops(node)
        # prune before pulling up order to avoid unnnecessary row_number() ops
        node = rewrites.column_pruning(node)
        node, ordering = rewrites.pull_up_order(node, order_root=ordered)
        # final pruning to cleanup up any leftovers unused values
        node = rewrites.column_pruning(node)
        # return self.compile_node(node).to_sql(
        #     order_by=ordering.all_ordering_columns if ordered else (),
        #     limit=limit,
        #     selections=output_ids,
        # )

        select_node = self.compile_node(node)

        order_expr = self.compile_row_ordering(ordering)
        if order_expr:
            select_node = select_node.order_by(order_expr)

        # return select_node

    def _replace_unsupported_ops(self, node: nodes.BigFrameNode) -> nodes.BigFrameNode:
        # TODO: Run all replacement rules as single bottom-up pass
        node = nodes.bottom_up(node, rewrites.rewrite_slice)
        node = nodes.bottom_up(node, rewrites.rewrite_timedelta_expressions)
        return node

    def compile_row_ordering(
        self, node: bigframes.core.ordering.RowOrdering
    ) -> sge.Order:
        if len(node.all_ordering_columns) == 0:
            return None

        ordering_expr = [
            sge.Ordered(
                this=sge.Column(
                    this=sge.to_identifier(
                        col_ref.scalar_expression.id.sql, quoted=self.quoted
                    )
                ),
                nulls_first=not col_ref.na_last,
                desc=not col_ref.direction.is_ascending,
            )
            for col_ref in node.all_ordering_columns
        ]
        return sge.Order(expressions=ordering_expr)

    @functools.singledispatchmethod
    def compile_node(self, node: nodes.BigFrameNode) -> sge.Select:
        """Defines transformation but isn't cached, always use compile_node instead"""
        raise ValueError(f"Can't compile unrecognized node: {node}")

    @compile_node.register
    def compile_selection(self, node: nodes.SelectionNode) -> sge.Select:
        child = self.compile_node(node.child)
        selected_cols = [
            sge.Alias(
                this=self.scalar_op_compiler.compile_expression(expr),
                alias=sge.to_identifier(id.name, quoted=self.quoted),
            )
            for expr, id in node.input_output_pairs
        ]
        return child.select(*selected_cols, append=False)

    @compile_node.register
    def compile_projection(self, node: nodes.ProjectionNode) -> sge.Select:
        child = self.compile_node(node.child)

        new_cols = [
            sge.Alias(
                this=self.scalar_op_compiler.compile_expression(expr),
                alias=sge.to_identifier(id.name, quoted=self.quoted),
            )
            for expr, id in node.assignments
        ]

        return child.select(*new_cols, append=True)

    @compile_node.register
    def compile_readlocal(self, node: nodes.ReadLocalNode) -> sge.Select:
        array_as_pd = pd.read_feather(
            io.BytesIO(node.feather_bytes),
            columns=[item.source_id for item in node.scan_list.items],
        )
        scan_list_items = node.scan_list.items

        # In the order mode, adds the offset column containing the index (0 to N-1)
        if node.offsets_col:
            offsets_col_name = node.offsets_col.sql
            array_as_pd[offsets_col_name] = range(len(array_as_pd))
            scan_list_items = scan_list_items + (
                nodes.ScanItem(
                    ids.ColumnId(offsets_col_name), dtypes.INT_DTYPE, offsets_col_name
                ),
            )

        # Convert timedeltas to microseconds for compatibility with BigQuery
        _ = utils.replace_timedeltas_with_micros(array_as_pd)

        array_expr = sge.DataType(
            this=sge.DataType.Type.STRUCT,
            expressions=[
                sge.ColumnDef(
                    this=sge.to_identifier(item.source_id, quoted=self.quoted),
                    kind=sgt.SQLGlotType.from_bigframes_dtype(item.dtype),
                )
                for item in scan_list_items
            ],
            nested=True,
        )
        array_values = [
            sge.Tuple(
                expressions=tuple(
                    self.literal(
                        value=value,
                        dtype=sgt.SQLGlotType.from_bigframes_dtype(item.dtype),
                    )
                    for value, item in zip(row, scan_list_items)
                )
            )
            for _, row in array_as_pd.iterrows()
        ]
        expr = sge.Unnest(
            expressions=[
                sge.DataType(
                    this=sge.DataType.Type.ARRAY,
                    expressions=[array_expr],
                    nested=True,
                    values=array_values,
                ),
            ],
        )
        return sg.select(sge.Star()).from_(expr)

    @compile_node.register
    def compile_filter(self, node: nodes.FilterNode) -> sge.Select:
        child_expr = self.compile_node(node.child)
        # cte_name = self.uid_generator.generate_sequential_uid("t")
        # with_expr = self.create_cte_from_select(child_expr, cte_name)

        # predicate_expr = self.scalar_op_compiler.compile_expression(node.predicate)

        # result = (
        #     sg.select(sge.Star())
        #     .from_(sg.to_identifier(cte_name, quoted=self.quoted))
        #     .where(predicate_expr)
        # )
        # existing = result.args.get("with")
        # if not existing:
        #     result.args.set("with", sge.With())
        # result.args.get("with").expressions = cte_list

        # predicate = node.predicate
        # def filter(self, predicate: ex.Expression) -> UnorderedIR:
        #     table = self._to_ibis_expr()
        #     condition = op_compiler.compile_expression(predicate, table)
        #     table = table.filter(condition)
        #     return UnorderedIR(
        #         table, tuple(table[column_name] for column_name in self._column_names)
        #     )

        return child_expr

    # TODO(refactor): Helpers to build SQLGlot expressions.
    def cast(self, arg, to) -> sge.Cast:
        return sge.Cast(this=sge.convert(arg), to=to, copy=False)

    def literal(self, value, dtype) -> sge.Expression:
        if value is None:
            return self.cast(sge.Null(), dtype)

        # TODO: handle other types like visit_DefaultLiteral
        return sge.convert(value)

    def create_cte_from_select(
        self, select: sge.Select, cte_name: str
    ) -> sge.With:
        new_cte = sge.CTE(
            this=select,
            alias=sge.TableAlias(this=sg.to_identifier(cte_name, quoted=self.quoted)),
        )

        with_expr = select.args.pop("with", sge.With())
        cte_list = with_expr.expressions + [new_cte]
        return sge.With(expressions=cte_list)
