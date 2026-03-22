# Copyright 2026 Google LLC
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
from typing import Type, TYPE_CHECKING

import pandas as pd
import pyarrow as pa

import bigframes.core
from bigframes.core import agg_expressions, nodes
import bigframes.core.expression as ex
import bigframes.dtypes
import bigframes.operations as ops

datafusion_installed = True
if TYPE_CHECKING:
    import datafusion
else:
    try:
        import bigframes._importing

        datafusion = bigframes._importing.import_datafusion()
    except Exception:
        datafusion_installed = False


def register_op(op: Type):
    """Register a compilation from BigFrames to DataFusion.

    This decorator can be used, even if DataFusion is not installed.

    Args:
        op: The type of the operator the wrapped function compiles.
    """

    def decorator(func):
        if datafusion_installed:
            return DataFusionExpressionCompiler.compile_op.register(op)(func)  # type: ignore
        else:
            return func

    return decorator


if datafusion_installed:
    _DTYPE_MAPPING = {
        bigframes.dtypes.INT_DTYPE: pa.int64(),
        bigframes.dtypes.FLOAT_DTYPE: pa.float64(),
        bigframes.dtypes.BOOL_DTYPE: pa.bool_(),
        bigframes.dtypes.STRING_DTYPE: pa.string(),
        # For now, map numeric to double or decimal if supported
        bigframes.dtypes.NUMERIC_DTYPE: pa.decimal128(38, 9),
        bigframes.dtypes.BIGNUMERIC_DTYPE: pa.decimal256(76, 38),
        bigframes.dtypes.BYTES_DTYPE: pa.binary(),
        bigframes.dtypes.DATE_DTYPE: pa.date32(),
        bigframes.dtypes.DATETIME_DTYPE: pa.timestamp("us"),
        bigframes.dtypes.TIMESTAMP_DTYPE: pa.timestamp("us", tz="UTC"),
        bigframes.dtypes.TIME_DTYPE: pa.time64("us"),
        bigframes.dtypes.TIMEDELTA_DTYPE: pa.duration("us"),
        bigframes.dtypes.GEO_DTYPE: pa.string(),
        bigframes.dtypes.JSON_DTYPE: pa.string(),
    }

    def _bigframes_dtype_to_arrow_dtype(
        dtype: bigframes.dtypes.ExpressionType,
    ) -> pa.DataType:
        if dtype is None:
            return pa.null()
        # TODO: Add struct and array handling if needed
        return _DTYPE_MAPPING[dtype]

    @dataclasses.dataclass(frozen=True)
    class DataFusionExpressionCompiler:
        """
        Compiler for converting bigframes expressions to datafusion expressions.
        """

        @functools.singledispatchmethod
        def compile_expression(self, expression: ex.Expression) -> datafusion.Expr:
            raise NotImplementedError(f"Cannot compile expression: {expression}")

        @compile_expression.register
        def _(
            self,
            expression: ex.ScalarConstantExpression,
        ) -> datafusion.Expr:
            value = expression.value
            if not isinstance(value, float) and pd.isna(value):  # type: ignore
                value = None
            if expression.dtype is None:
                return datafusion.lit(None)

            # DataFusion lit handles standard types
            return datafusion.lit(value)

        @compile_expression.register
        def _(
            self,
            expression: ex.DerefOp,
        ) -> datafusion.Expr:
            return datafusion.col(expression.id.sql)

        @compile_expression.register
        def _(
            self,
            expression: ex.ResolvedDerefOp,
        ) -> datafusion.Expr:
            return datafusion.col(expression.id.sql)

        @compile_expression.register
        def _(
            self,
            expression: ex.OpExpression,
        ) -> datafusion.Expr:
            op = expression.op
            args = tuple(map(self.compile_expression, expression.inputs))
            return self.compile_op(op, *args)

        @functools.singledispatchmethod
        def compile_op(
            self, op: ops.ScalarOp, *args: datafusion.Expr
        ) -> datafusion.Expr:
            raise NotImplementedError(f"DataFusion compiler hasn't implemented {op}")

        # Add basic ops here, others via register_op
        # df expressions overload operators usually

    @dataclasses.dataclass(frozen=True)
    class DataFusionAggregateCompiler:
        scalar_compiler = DataFusionExpressionCompiler()

        def compile_agg_expr(self, expr: agg_expressions.Aggregation):
            # Skeleton for now
            raise NotImplementedError("Aggregate compilation not implemented")

    @dataclasses.dataclass(frozen=True)
    class DataFusionCompiler:
        """
        Compiles BigFrameNode to DataFusion DataFrame.
        """

        expr_compiler = DataFusionExpressionCompiler()
        agg_compiler = DataFusionAggregateCompiler()

        def compile(self, plan: nodes.BigFrameNode) -> datafusion.DataFrame:
            if not datafusion_installed:
                raise ValueError(
                    "DataFusion is not installed, cannot compile to datafusion engine."
                )

            from bigframes.core.compile.datafusion import lowering

            node = lowering.lower_ops_to_datafusion(plan)
            return self.compile_node(node)

        @functools.singledispatchmethod
        def compile_node(self, node: nodes.BigFrameNode) -> datafusion.DataFrame:
            raise ValueError(f"Can't compile unrecognized node: {node}")

        @compile_node.register
        def compile_readlocal(self, node: nodes.ReadLocalNode):
            # Need SessionContext, maybe pass it in or create one
            ctx = datafusion.SessionContext()
            df = ctx.from_arrow(node.local_data_source.data)

            cols_to_read = {
                scan_item.source_id: scan_item.id.sql
                for scan_item in node.scan_list.items
            }
            # Rename columns
            # DataFusion select can take list of expressions
            exprs = [
                datafusion.col(orig).alias(new) for orig, new in cols_to_read.items()
            ]
            df = df.select(*exprs)

            if node.offsets_col:
                # DataFusion has row_number()?
                # But ReadLocalNode usually has small data, could just use arrow offsets if needed
                # For now, let's just make it error if offsets_col is requested and see
                raise NotImplementedError(
                    "offsets_col in ReadLocalNode not supported yet for DataFusion"
                )
            return df

        @compile_node.register
        def compile_filter(self, node: nodes.FilterNode):
            return self.compile_node(node.child).filter(
                self.expr_compiler.compile_expression(node.predicate)
            )

        @compile_node.register
        def compile_selection(self, node: nodes.SelectionNode):
            df = self.compile_node(node.child)
            exprs = [
                datafusion.col(orig.id.sql).alias(new.sql)
                for orig, new in node.input_output_pairs
            ]
            return df.select(*exprs)

        @compile_node.register
        def compile_projection(self, node: nodes.ProjectionNode):
            df = self.compile_node(node.child)
            new_cols = []
            for proj_expr, name in node.assignments:
                # bind_schema_fields might be needed
                bound_expr = ex.bind_schema_fields(proj_expr, node.child.field_by_id)
                new_col = self.expr_compiler.compile_expression(bound_expr).alias(
                    name.sql
                )
                new_cols.append(new_col)

            # with_columns takes dict or list of aliases?
            # DF DataFrame has with_column
            for col in new_cols:
                # df = df.with_column(col) # wait, with_column usually takes name and expr
                # let's see df.select(*existing, new)
                pass
            # Better to use select with existing columns + new columns
            new_names = [name.sql for _, name in node.assignments]
            filtered_existing = [
                datafusion.col(c) for c in df.schema().names if c not in new_names
            ]
            return df.select(*(filtered_existing + new_cols))
