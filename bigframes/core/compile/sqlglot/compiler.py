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
from typing import cast, Sequence, Tuple, TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import sqlglot as sg
import sqlglot.expressions as sge

import bigframes.core
import bigframes.core.compile.sqlglot.sqlglot_types as sgt
import bigframes.core.expression as ex
import bigframes.core.guid as guid
import bigframes.core.nodes as nodes
import bigframes.core.rewrite
import bigframes.dtypes as dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops


@dataclasses.dataclass(frozen=True)
class SQLGlotCompiler:
    """
    Compiles BigFrameNode to SQLGlot expression tree recursively.
    """

    # Whether to always quote identifiers.
    quoted: bool = True

    # TODO: add BigQuery Dialect

    def compile(self, array_value: bigframes.core.ArrayValue) -> sg.Expression:
        # TODO: do we need rewrite here?
        # node = nodes.bottom_up(array_value.node, bigframes.core.rewrite.rewrite_slice)
        return self.compile_node(array_value.node)

    @functools.singledispatchmethod
    def compile_node(self, node: nodes.BigFrameNode):
        """Defines transformation but isn't cached, always use compile_node instead"""
        raise ValueError(f"Can't compile unrecognized node: {node}")

    @compile_node.register
    def compile_selection(self, node: nodes.SelectionNode):
        return self.compile_node(node.child)

    @compile_node.register
    def compile_readlocal(self, node: nodes.ReadLocalNode):
        array_as_pd = pd.read_feather(
            io.BytesIO(node.feather_bytes),
            columns=[item.source_id for item in node.scan_list.items],
        )

        quoted = self.quoted
        array_expr = sge.DataType(
            this=sge.DataType.Type.STRUCT,
            expressions=[
                sge.ColumnDef(
                    this=sge.to_identifier(item.source_id, quoted=quoted),
                    kind=sgt.SQLGlotType.from_bigframes_dtype(item.dtype),
                )
                for item in node.scan_list.items
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
                    for value, item in zip(row, node.scan_list.items)
                )
            )
            for _, row in array_as_pd.iterrows()
        ]
        # TODO: check table alias
        name = "table_alias"
        expr = sge.Unnest(
            expressions=[
                sge.DataType(
                    this=sge.DataType.Type.ARRAY,
                    expressions=[array_expr],
                    nested=True,
                    values=array_values,
                ),
            ],
            alias=sge.TableAlias(
                columns=[sg.to_identifier(name, quoted=quoted)],
            ),
        )
        return sg.select(sge.Star()).from_(expr)

    def cast(self, arg, to) -> sge.Cast:
        return sge.Cast(this=sge.convert(arg), to=to, copy=False)

    def literal(self, value, dtype) -> sge.Expression:
        if value is None:
            return self.cast(sge.Null(), dtype)

        # TODO: handle other types like visit_DefaultLiteral
        return sge.convert(value)
